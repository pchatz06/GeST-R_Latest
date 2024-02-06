'''
Copyright 2019 ARM Ltd. and University of Cyprus
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, 
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

from Measurement.Measurement import Measurement
import math
class MeasurementLikwidPower(Measurement):
    '''
    classdocs
    '''
    
    def __init__(self,confFile):
        super().__init__(confFile)
     
    
    def init(self):
        super().init()
        self.timeToMeasure = self.tryGetIntValue('time_to_measure')
    
    #the below code is tested to work with likwid 4.3 and 3.1 versions
    def euclidean(self, individual, features):
        measurements = []
        # super().copyFileOverFTP()
        # compilation_command = "cd " + self.targetRunDir + " ; gcc main.s -o individual &>/dev/null;"
        # execution_command = "cd " + self.targetRunDir + " ; "
        # super().executeSSHcommand(compilation_command)
        # super().executeSSHcommand(execution_command)

        # print(f"Features: {features}")

        mload = features[0][0]
        mstore = features[1][0]
        mscalar = features[2][0]
        mscalar_ls = features[3][0]
        mvmul = features[4][0]
        mvadd = features[5][0]
        mvmax = features[6][0]
        mvsub = features[7][0]
        mvxor = features[8][0]
        mtot = mload + mstore + mscalar + mscalar_ls + mvmul + mvadd + mvmax + mvsub + mvxor

        mload = mload / mtot
        mstore = mstore / mtot
        mscalar = mscalar / mtot
        msls = mscalar_ls / mtot
        mvmultpd = mvmul / mtot
        mvaddpd = mvadd / mtot
        mvmaxpd = mvmax / mtot
        mvsubpd = mvsub / mtot
        mbxorpd = mvxor / mtot

        load = 0
        store = 0
        scalar = 0
        scalar_ls = 0
        vmul = 0
        vadd = 0
        vmax = 0
        vsub = 0
        vxor = 0
        ls = 0

        for ins in individual.sequence:
            if ((ins.name == "ADD") or (ins.name == "ADD_IM") or (ins.name == "MUL_IM") or (ins.name == "MUL") or (
                    ins.name == "SHL") or (ins.name == "SAR") or (ins.name == "ROR") or (ins.name == "CMP")):
                scalar += 1
            if (ins.name == "ADD_2ndMem"):
                scalar_ls += 1
            if ((ins.name == "MOV_2ndMem") or (ins.name == "MOV_1stMem") or (ins.name == "MOV")):
                ls += 1
                if (ins.name == "MOV_1stMem"):
                    load += 1
                elif (ins.name == "MOV_2ndMem"):
                    store += 1
                else:
                    store += 1
            if ((ins.name == "AtomicMovSeq1") or (ins.name == "AtomicMovSeq2") or (ins.name == "AtomicMovSeq3")):
                ls += 3
                load += 3
            if (ins.name == "VADDPD"):
                vadd += 1
            if (ins.name == "VMULPD"):
                vmul += 1
            if (ins.name == "VMAXPD"):
                vmax += 1
            if (ins.name == "VSUBPD"):
                vsub += 1
            if (ins.name == "VXORPD"):
                vxor += 1

        tot = scalar + scalar_ls + load + store + vmul + vadd + vxor + vsub + vmax
        load1 = load / tot
        store1 = store / tot
        scalar1 = scalar / tot
        scalar_ls1 = scalar_ls / tot
        vadd1 = vadd / tot
        vmul1 = vmul / tot
        vmax1 = vmax / tot
        vsub1 = vsub / tot
        vxor1 = vxor / tot
        tot1 = abs(1 - (tot / mtot))

        euclidan = math.sqrt(
            math.pow(scalar1 - mscalar, 2) + math.pow(load1 - mload, 2) + math.pow(store1 - mstore, 2) +
            math.pow(vadd1 - mvaddpd, 2) + math.pow(
                vmul1 - mvmultpd, 2) + math.pow(vmax1 - mvmaxpd, 2) + math.pow(vsub1 - mvsubpd, 2) + math.pow(
                scalar_ls1 - msls, 2) + math.pow(vxor1 - mbxorpd, 2) + math.pow(tot1, 2))
        euclidan = -euclidan

        measurements.append(euclidan)
        measurements.append(load)
        measurements.append(store)
        measurements.append(scalar)
        measurements.append(scalar_ls)
        measurements.append(vmul)
        measurements.append(vadd)
        measurements.append(vmax)
        measurements.append(vsub)
        measurements.append(vxor)
        measurements.append(tot)

        return measurements

    def measure(self):  
            
        super().copyFileOverFTP()
        compilation_command="cd "+self.targetRunDir + " ; gcc main.s -o individual &>/dev/null;"
        execution_command="cd "+self.targetRunDir+" ; "
        for core in self.coresToUse:
                execution_command+="taskset -c "+str(core)+" ./individual  &>/dev/null &  "
        execution_command+=" sudo likwid-powermeter  -s "+str(self.timeToMeasure) +"s > tmp ; pkill individual &> /dev/null;" #make sure that msr module is loaded (modprobe msr) and sudo without password is enabled
        output_command="cd "+self.targetRunDir + " ; cat tmp | grep Watt | head -n 1 | awk '{print $3}'; rm main.s; rm individual; rm tmp; "; #this grabs the package power
        super().executeSSHcommand(compilation_command)
        super().executeSSHcommand(execution_command)
        stdout=super().executeSSHcommand(output_command)
                     
        
        for line in stdout:
            try:
                test=float(line)
                power_meas=test
            except ValueError:
                print ("Exception line not power")
   

        measurements=[];
        measurements.append(power_meas);
        
        return measurements;
            