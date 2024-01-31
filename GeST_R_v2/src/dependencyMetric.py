import copy
class ParsedInstruction:
    def __init__(self, insType, readRegs, writeReg, string):
        self.instruction_type = insType
        self.read_registers = readRegs
        self.write_register = writeReg
        self.executed = "NO"
        self.string = string

    def get_write_register(self):
        return self.write_register

    def get_read_registers(self):
        return self.read_registers

    def get_string(self):
        return self.string

    def set_string(self, string):
        self.string = string

    def get_instruction_type(self):
        return self.instruction_type

    def __str__(self):
        return (
            f"{self.string}\n{self.instruction_type} read:{','.join(str(read_reg) for read_reg in self.read_registers)},"
            f"write:{self.write_register}")


def parse_instructions(individual):
    instructions = individual.split("\n")
    instructions = [line.replace("\t", "") for line in instructions if line != ""]

    parsed_instructions = []
    for instruction in instructions:
        if len(instruction.strip()) > 0:
            ins = instruction.split(' ')
            instruction_type = ins[0]

            instruction_read = []
            ins = ins[1]
            ins = ins.split(',')
            instruction_write = "None"
            if instruction_type != "cmp":
                if '(' not in ins[len(ins) - 1]:
                    instruction_write = ins[len(ins) - 1]
                else:
                    instruction_write = "None"
                    instruction_read.append(
                        ins[len(ins) - 1].split('(')[1].split(')')[0])  # FIX HERE comment this out!!!

            if instruction_type == "mov" or "%ymm" in instruction_write or "%zmm" in instruction_write or "%xmm" in instruction_write:
                for i in range(0, len(ins) - 1):
                    if "$" not in ins[i]:
                        if '(' in ins[i]:
                            ins[i] = ins[i].split('(')[1].split(')')[0]
                        instruction_read.append(ins[i])
            else:
                for i in range(0, len(ins)):
                    if "$" not in ins[i]:
                        if '(' in ins[i]:
                            ins[i] = ins[i].split('(')[1].split(')')[0]
                        instruction_read.append(ins[i])

            parsed_instruction = ParsedInstruction(instruction_type, instruction_read, instruction_write, instruction)
            parsed_instructions.append(parsed_instruction)

        # print(parsed_instruction.get_string()) # debugging
    return parsed_instructions

def dependency_metric(individual, iterations):
    last_write_instruction = []
    write_registers = []
    instructions = parse_instructions(individual)
    vectors = 0
    other_ins = 0

    temp_instr = copy.deepcopy(instructions)

    for i in range(int(iterations)):
        instructions.extend(temp_instr)

    dependencies = [[-1 for _ in range(2)] for _ in range(int(len(instructions)))]
    latency = [[0 for _ in range(2)] for _ in range(int(len(instructions)))]

    dependency_distance = 0
    third_iteration = []
    all_iterations = []
    RAW_Vectors = 0
    RAW = 0
    dependency_list = []


    for i in range(len(instructions)):
        fetch_score = int(i / 4)
        current_loop = int(i / len(temp_instr)) + 1
        write_register = instructions[i].get_write_register()
        read_registers = instructions[i].get_read_registers()
        instruction_type = instructions[i].get_instruction_type()
        if instruction_type == "vmulpd":
            latency[i][0] = 2
        else:
            latency[i][0] = 1

        read_register_id = 0
        for read_register in read_registers:
            if read_register in write_registers:
                if current_loop == iterations + 1 and instruction_type[0] == 'v':
                    RAW_Vectors += 1
                elif current_loop == iterations + 1 and instruction_type[0] != 'v':
                    RAW += 1

                index = write_registers.index(read_register)
                instruction_id = last_write_instruction[index]
                dependencies[i][read_register_id] = instruction_id

                if current_loop == iterations + 1:
                    dependency_distance += i - instruction_id

                if instruction_type == "vmulpd":
                    latency[i][read_register_id] = max(latency[instruction_id]) + 2
                else:
                    latency[i][read_register_id] = max(latency[instruction_id]) + 1

                read_register_id += 1

        if write_register != "None":
            if write_register not in write_registers:
                write_registers.append(write_register)
                last_write_instruction.append(i)
            else:
                index = write_registers.index(write_register)
                last_write_instruction[index] = i

        if current_loop == iterations + 1:
            if instruction_type[0] == 'v':
                vectors += 1
            else:
                other_ins += 1
            third_iteration.append(max(latency[i]))
        all_iterations.append(max(latency[i]))

        current_list = []
        current_list.append(dependencies[i])
        current_list.append(instructions[i].get_string())
        current_list.append(latency[i])
        current_list.append(current_loop)
        dependency_list.append(current_list)

    latency_depth_all_iterations = [[] * 10000 for _ in range(int(len(instructions) + 10000))]
    latency_depth_third_iteration = [0 for _ in range(int(max(third_iteration)) + 1)]
    for i in range(len(latency_depth_third_iteration)):
        for j in range(len(third_iteration)):
            if third_iteration[j] == i:
                latency_depth_third_iteration[i] += 1

    max_latency = 0
    for i in range(len(dependency_list)):
        latency = max(dependency_list[i][2])
        if latency > max_latency:
            max_latency = latency
        latency_depth_all_iterations[latency].append(i)

    metric = 0
    for d in range(len(latency_depth_third_iteration)):
        metric += latency_depth_third_iteration[d] * d

    # print_graph(latency_depth_all_iterations, dependency_list)
    # return metric / math.pow(RAW, 5) if RAW != 0 else metric # old
    return [metric,
            len(latency_depth_third_iteration),
            vectors,
            other_ins,
            RAW_Vectors,
            RAW,
            (metric / vectors) if vectors != 0 else metric]