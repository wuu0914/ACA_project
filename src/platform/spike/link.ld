OUTPUT_ARCH("riscv")
ENTRY(_start)

/* Spike's default DRAM begins at 0x80000000; Phase 2 runs it with -m16. */
MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 16M
}

PHDRS
{
    text PT_LOAD FLAGS(5); /* read + execute */
    data PT_LOAD FLAGS(6); /* read + write */
}

SECTIONS
{
    . = ORIGIN(RAM);

    /* Startup comes first so _start is also the first target instruction. */
    .text :
    {
        KEEP(*(.text.init))
        *(.text .text.*)
    } > RAM :text

    /* Model weights, generated test input and golden output are read-only. */
    .rodata :
    {
        *(.rodata .rodata.*)
    } > RAM :text

    .data :
    {
        *(.data .data.*)
        PROVIDE(__global_pointer$ = . + 0x800);
    } > RAM :data

    /* Keep initialized HTIF mailboxes before NOLOAD sections in the data segment. */
    .tohost :
    {
        KEEP(*(.tohost))
    } > RAM :data

    /* Zero-initialized model output, tensor arena and debug result variables. */
    .bss (NOLOAD) :
    {
        . = ALIGN(8);
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(8);
        __bss_end = .;
    } > RAM :data

    /* No function uses a large local array; 4 KiB is enough for this call stack. */
    .stack (NOLOAD) :
    {
        . = ALIGN(16);
        . += 4096;
        __stack_top = .;
    } > RAM :data

    /* Fail at link time instead of silently placing data outside Spike RAM. */
    ASSERT(__stack_top <= ORIGIN(RAM) + LENGTH(RAM),
           "efficientnet_spike.elf exceeds 16 MiB RAM")

    /DISCARD/ :
    {
        *(.comment)
        *(.note .note.*)
        *(.eh_frame .eh_frame.*)
    }
}
