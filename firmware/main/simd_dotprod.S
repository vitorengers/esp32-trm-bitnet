/**
 * ESP32-S3 SIMD INT8 Dot Product using PIE (Processor Instruction Extensions)
 *
 * Uses the Xtensa LX7 128-bit SIMD registers (q0-q7) and 40-bit accumulator (ACCX).
 *
 * Key instructions:
 *   ee.zero.accx           — Clear ACCX accumulator to 0
 *   ee.vld.128.ip qX, aY, 16  — Load 16 bytes into qX from [aY], post-increment by 16
 *   ee.vmulas.s8.accx qX, qY  — ACCX += sum of 16 signed 8-bit products (qX[i] * qY[i])
 *   ee.vmulas.s8.accx.ld.ip qZ, aW, 16, qX, qY
 *                              — Fused: load 16 bytes into qZ AND accumulate qX*qY
 *   ee.srs.accx aX             — Store ACCX result into general register aX
 *
 * Function signature (called from C):
 *   int32_t simd_dotprod_s8(const int8_t *a, const int8_t *b, int len);
 *
 * Parameters:
 *   a2 = pointer to first INT8 array (must be 16-byte aligned)
 *   a3 = pointer to second INT8 array (must be 16-byte aligned)
 *   a4 = number of elements (must be multiple of 16)
 *
 * Returns:
 *   a2 = dot product result (int32_t, lower 32 bits of ACCX)
 *
 * Performance: ~1 cycle per 16 elements (fused load + MAC)
 */

    .text
    .align 4
    .global simd_dotprod_s8
    .type   simd_dotprod_s8, @function

simd_dotprod_s8:
    entry   a1, 32                  /* Create stack frame */

    /* Clear accumulator */
    ee.zero.accx

    /* a4 = len, compute number of 16-element iterations */
    srli    a5, a4, 4               /* a5 = len / 16 = number of iterations */

    /* Load first pair of vectors */
    ee.vld.128.ip   q0, a2, 16     /* q0 = a[0..15], a2 += 16 */
    ee.vld.128.ip   q1, a3, 16     /* q1 = b[0..15], a3 += 16 */

    addi    a5, a5, -1              /* Decrement loop count (first pair already loaded) */
    beqz    a5, .Llast_mac          /* If only 1 iteration, jump to final MAC */

    /* Main loop: fused load + MAC
     * Each iteration:
     *   1. Accumulate previous pair: ACCX += sum(q0[i]*q1[i])
     *   2. Load next pair into q0, q1
     */
.Lloop:
    /* MAC q0*q1 and load next a[] into q0 */
    ee.vmulas.s8.accx.ld.ip  q0, a2, 16, q0, q1
    /* Load next b[] into q1 */
    ee.vld.128.ip   q1, a3, 16

    addi    a5, a5, -1
    bnez    a5, .Lloop

.Llast_mac:
    /* Final MAC for the last loaded pair */
    ee.vmulas.s8.accx   q0, q1

    /* Read accumulator result (shift by 0 = raw value) */
    movi    a5, 0                   /* a5 = shift amount = 0 */
    ee.srs.accx   a2, a5, 0        /* a2 = (int32_t) ACCX >> 0 */

    retw                            /* Return from windowed call */

    .size   simd_dotprod_s8, . - simd_dotprod_s8
