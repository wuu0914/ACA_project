    /* Startup code must be the first code placed at the ELF entry point. */
    .section .text.init
    .globl _start
    .type _start, @function

_start:
    /* Initialize gp without allowing linker relaxation to reuse an unset gp. */
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop

    /* The linker script provides a 16-byte-aligned, downward-growing stack. */
    la sp, __stack_top

    /*
     * ELF does not store the bytes of a NOLOAD .bss section. Clear it before
     * C code runs so the tensor arena and all zero-initialized globals start at 0.
     * link.ld aligns both boundaries, so each iteration may store eight bytes.
     */
    la t0, __bss_start
    la t1, __bss_end
    bgeu t0, t1, 2f
1:
    sd zero, 0(t0)
    addi t0, t0, 8
    bltu t0, t1, 1b

    /* main() returns 0 for PASS and a nonzero value for failure. */
2:
    call main

    /* Spike HTIF exit message: (exit_status << 1) | 1. */
    slli a0, a0, 1
    ori a0, a0, 1
    la t0, tohost
    sd a0, 0(t0)

    /* Wait safely if a simulator does not consume the HTIF message. */
3:
    wfi
    j 3b

    .size _start, .-_start

    /* Spike locates these 64-byte-aligned HTIF mailboxes by ELF symbol name. */
    .section .tohost, "aw", @progbits
    .balign 64
    .globl tohost
tohost:
    .dword 0

    .balign 64
    .globl fromhost
fromhost:
    .dword 0
