/*
Copyright 2019 ARM Ltd. and University of Cyprus
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, 
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

        .file   "main.s"
        .data
        .align 32
        .simdvalue:
            .long   0xaaaaaaaa
            .long   0x55555555
            .long   0x33333333
            .long   0xcccccccc
            .long   0xaaaaaaaa
            .long   0x55555555
            .long   0x33333333
            .long   0xcccccccc
        .text
        .globl  main
        main:
.LFB0:
        .cfi_startproc
        pushq   %rbp
        .cfi_def_cfa_offset 8
        .cfi_offset 5, -8
        movl    %esp, %ebp
        .cfi_def_cfa_register 5



        #reg init


        mov $0x55555555, %rax
        mov $0x33333333, %rbx
        mov $0x22222222, %rdx
        mov $0x44444444, %rsi
        mov $0x77777777, %rdi

        fldpi
        fldpi
        fldpi
        fldpi
        fldpi
        fldpi
        fldpi
        
        vmovdqa .simdvalue(%rip), %ymm0
        vmovdqa .simdvalue(%rip), %ymm1
        vmovdqa .simdvalue(%rip), %ymm2
        vmovdqa .simdvalue(%rip), %ymm3
        vmovdqa .simdvalue(%rip), %ymm4
        vmovdqa .simdvalue(%rip), %ymm5
        vmovdqa .simdvalue(%rip), %ymm6
        vmovdqa .simdvalue(%rip), %ymm7
        vmovdqa .simdvalue(%rip), %ymm8
        vmovdqa .simdvalue(%rip), %ymm9
        vmovdqa .simdvalue(%rip), %ymm10
        vmovdqa .simdvalue(%rip), %ymm11
        vmovdqa .simdvalue(%rip), %ymm12
        vmovdqa .simdvalue(%rip), %ymm13
        vmovdqa .simdvalue(%rip), %ymm14
        vmovdqa .simdvalue(%rip), %ymm15

        mov $50000000, %rcx  #leave for i--

        #subq    $304, %rsp

.L2:
      
	mov 0(%rsp),%rbx
	mov 64(%rsp),%rdx
	mov 128(%rsp),%rsi
	add %rdi,%rbx
	mov %rdx,56(%rsp)
	mov 384(%rsp),%rsi
	mov 448(%rsp),%rdx
	mov 512(%rsp),%rbx
	vmulpd %ymm4,%ymm1,%ymm5
	vmaxpd %ymm10,%ymm2,%ymm10
	vaddpd %ymm8,%ymm15,%ymm12
	add %rdx,%rdx
	vmaxpd %ymm1,%ymm7,%ymm0
	cmp %rdx,%rsi
	mov %rdi,%rbx
	vsubpd %ymm6,%ymm6,%ymm3
	cmp %rdi,%rax
	vaddpd %ymm10,%ymm10,%ymm4
	imul $286331140,%rsi
	vsubpd %ymm3,%ymm7,%ymm1
	vmaxpd %ymm13,%ymm0,%ymm8
	vmaxpd %ymm6,%ymm9,%ymm12
	add %rdi,%rdx
	vsubpd %ymm3,%ymm5,%ymm7
	vmulpd %ymm13,%ymm2,%ymm2
	mov %rsi,28(%rsp)
	ror $31,%rdx
	vmaxpd %ymm1,%ymm3,%ymm8
	ror $31,%rbx
	mov %rbx,%rsi
	sar $31,%rax
	sar $31,%rax
	add %rax,%rbx
	mov 384(%rsp),%rdi
	mov 448(%rsp),%rdi
	mov 512(%rsp),%rdx
	vsubpd %ymm14,%ymm6,%ymm4
	add $71582785,%rdi
	vmaxpd %ymm4,%ymm15,%ymm11
	vsubpd %ymm12,%ymm6,%ymm8
	vsubpd %ymm8,%ymm9,%ymm14
	mov 384(%rsp),%rdx
	mov 448(%rsp),%rax
	mov 512(%rsp),%rbx
	vsubpd %ymm5,%ymm0,%ymm2
	vmulpd %ymm8,%ymm12,%ymm3
	imul $787410635,%rdx
	add $1789569625,%rdi
	mov 192(%rsp),%rax
	mov 256(%rsp),%rdi
	mov 320(%rsp),%rdx
	cmp %rdx,%rdi
	mov 384(%rsp),%rsi
	mov 448(%rsp),%rdi
	mov 512(%rsp),%rdi
	vxorpd %ymm1,%ymm12,%ymm12
	ror $31,%rsi
	add %rsi,%rdx
	shl $31,%rax
	vmulpd %ymm11,%ymm4,%ymm14
	add $1503238485,%rsi
	mov 384(%rsp),%rbx
	mov 448(%rsp),%rbx
	mov 512(%rsp),%rax



        #sub $1,%rcx #remove this and below comment for fixed iterations
        #cmp $0, %rcx
        jmp     .L2

         leave
        .cfi_restore 5
        .cfi_def_cfa 4, 4
       ret

        .cfi_endproc
.LFE0:
        .ident  "GCC: (GNU) 6.4.0"
