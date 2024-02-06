import os
import pickle
from Population import Population


# class Features:
#     def __init__(self, fitness, ls, load, store, scalar, scalar_ls, vmul, vadd, vmax, vsub, vxor):
#         self.fitness = round(float(fitness), 6)
#         self.ls = ls
#         self.load = load
#         self.store = store
#         self.scalar = scalar
#         self.scalar_ls = scalar_ls
#         self.vmul = vmul
#         self.vadd = vadd
#         self.vmax = vmax
#         self.vsub = vsub
#         self.vxor = vxor


def parseGeneration(file):
    input = open(file, "rb")
    pop = pickle.load(input)
    input.close()
    best = pop.getFittest()

    ls = 0
    scalar = 0
    scalar_ls = 0
    vmul = 0
    vadd = 0
    vmax = 0
    vsub = 0
    vxor = 0
    load = 0
    store = 0

    for ins in best.sequence:
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

    features = [round(float(best.getFitness()), 6), load, store, scalar, scalar_ls, vmul, vadd, vmax, vsub, vxor]

    return features
