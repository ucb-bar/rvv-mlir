/* Full-output and refusal-atomicity comparison against the transposed Merlin reference. */
#include <fenv.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int merlin_ref(void *, const void *const *, void *const *);
int rvv_exact_bf16_linear(void *, const void *const *, void *const *);
/* Read architectural frm directly so a pk/newlib fenv stub cannot hide the
   non-RNE case. The leaves only compare against FE_TONEAREST; distinct synthetic
   values suffice for all other modes, including reserved frm encodings. */
int fegetround(void) {
  unsigned long r;
  __asm__ volatile("frrm %0" : "=r"(r));
  return r == 0 ? FE_TONEAREST : (int)r + 100;
}
static inline unsigned long instret(void) {
  unsigned long v;
  __asm__ volatile("rdinstret %0" : "=r"(v));
  return v;
}
static uint64_t state;
static uint64_t rnd(void) {
  state ^= state << 13; state ^= state >> 7; state ^= state << 17;
  return state;
}
static uint16_t gen(int mode) {
  uint16_t sign = (uint16_t)((rnd() & 1u) << 15), man = (uint16_t)(rnd() & 127u);
  int e;
  switch (mode) {
  case 0: e = 124 + (int)(rnd() % 7u); break;
  case 1:
    e = 57 + (int)(rnd() % 90u);
    if (rnd() % 16u == 0) e = 0;
    if (rnd() % 32u == 0) { e = 0; man = 0; }
    break;
  case 2: e = 120 + (int)(rnd() % 60u); break;
  case 3: e = (rnd() % 2u) ? 127 : 0; man = (rnd() % 4u) ? 0 : man; break;
  case 4: e = (int)(rnd() % 64u); break;
  case 5: e = 186 + (int)(rnd() % 8u); break;
  case 6: e = (rnd() % 64u) ? 126 : 186 + (int)(rnd() % 8u); break;
  default: e = 0; man = 0; break;
  }
  return (uint16_t)(sign | (e << 7) | man);
}

/* Exercise each guard before invalid addresses can reach an input load. The
   alias arena is large enough for either span even if a broken leaf writes. */
static int abi_refusals(const void *const *rin, const void *const *vin,
                        const size_t sizes[3], size_t bytes) {
  size_t largest = sizes[0];
  for (unsigned p = 1; p < 3; ++p) if (sizes[p] > largest) largest = sizes[p];
  const size_t alias_bytes = largest + bytes + 64, guard_bytes = bytes + 32;
  unsigned char *alias = malloc(alias_bytes), *guard = malloc(guard_bytes);
  if (!alias || !guard) { free(alias); free(guard); return 0; }
  int (*leaves[2])(void *, const void *const *, void *const *) = {
    merlin_ref, rvv_exact_bf16_linear
  };
  for (unsigned leaf = 0; leaf < 2; ++leaf) for (unsigned test = 0; test < 19; ++test) {
    const void *const *original = leaf ? vin : rin;
    const void *inputs[3] = {original[0], original[1], original[2]};
    void *outputs[1] = {guard + 16};
    memset(alias, 0xa5, alias_bytes); memset(guard, 0xcd, guard_bytes);
    if (test == 2) outputs[0] = 0;
    if (test >= 3 && test < 6) inputs[test - 3] = 0;
    if (test >= 6 && test < 9) {
      unsigned p = test - 6;
      inputs[p] = (const void *)(UINTPTR_MAX - sizes[p] + 1);
    }
    if (test == 9) outputs[0] = (void *)(UINTPTR_MAX - bytes + 1);
    if (test >= 10) {
      unsigned p = (test - 10) / 3, kind = (test - 10) % 3;
      inputs[p] = alias + 16;
      /* Identical base, output overlapping the input's final element, then
         output beginning before input and overlapping its first element. */
      outputs[0] = kind == 1 ? alias + 16 + sizes[p] - 2 : alias + 16;
      /* Keep the preceding-output case within the arena for every shape. */
      if (kind == 2) { inputs[p] = alias + bytes + 16; outputs[0] = alias + 18; }
    }
    int rc = leaves[leaf](0, test == 0 ? 0 : inputs, test == 1 ? 0 : outputs);
    size_t changed = 0;
    for (size_t i = 0; i < alias_bytes; ++i) changed += alias[i] != 0xa5;
    for (size_t i = 0; i < guard_bytes; ++i) changed += guard[i] != 0xcd;
    if (rc != -1 || changed) {
      fprintf(stderr, "ABI refusal failed leaf=%u case=%u rc=%d changed=%zu\n",
              leaf, test, rc, changed);
      free(alias); free(guard); return 0;
    }
  }
  free(alias); free(guard); return 1;
}

int main(int argc, char **argv) {
  if (argc != 4) return 2;
  int mode = atoi(argv[1]);
  state = 0x9E3779B97F4A7C15ULL ^ (uint64_t)atoi(argv[2]);
  const size_t M = M_, K = K_, N = N_, bytes = 2 * M * N;
  uint16_t *a = malloc(M * K * 2), *w = malloc(N * K * 2), *wt = malloc(K * N * 2);
  unsigned char *reference = malloc(bytes + 32), *actual = malloc(bytes + 32);
  float *z = calloc(M * N, 4);
  if (!a || !w || !wt || !reference || !actual || !z) return 2;
  for (size_t i = 0; i < M * K; i++) a[i] = gen(mode);
  for (size_t i = 0; i < N * K; i++) w[i] = gen(mode);
  if (mode == 12 || mode == 13) {
    /* Exact f32 products halfway between BF16 neighbors, even and odd ties. */
    memset(a, 0, M * K * 2); memset(w, 0, N * K * 2);
    for (size_t m = 0; m < M; m++) a[m * K] = 0x3fc0;
    for (size_t n = 0; n < N; n++) w[n * K] = mode == 12 ? 0x3f83 : 0x3f81;
  }
  if (mode == 8) a[0] = 0x7fc1; /* NaN input */
  if (mode == 9) w[0] = 0x7f80; /* infinity input */
  if (mode == 10) z[0] = -0.0f; /* zero seed must be positive */
  if (mode == 11) __asm__ volatile("fsrmi 1"); /* reject non-RNE */
  for (size_t n = 0; n < N; n++) for (size_t k = 0; k < K; k++) wt[k * N + n] = w[n * K + k];
  memset(reference, 0xcd, bytes + 32); memset(actual, 0xcd, bytes + 32);
  const void *rin[3] = {a, wt, z}, *vin[3] = {a, w, z};
  void *rout[1] = {reference + 16}, *vout[1] = {actual + 16};
  const size_t sizes[3] = {2 * M * K, 2 * N * K, 4 * M * N};
  if (mode == 14 && !abi_refusals(rin, vin, sizes, bytes)) return 1;
  unsigned long t0 = instret();
  int rr = merlin_ref(0, rin, rout);
  unsigned long t1 = instret();
  int vr = rvv_exact_bf16_linear(0, vin, vout);
  unsigned long t2 = instret(), vlenb;
  __asm__ volatile("csrr %0,vlenb" : "=r"(vlenb));
  size_t diff = 0, sub = 0, nz = 0;
  for (size_t i = 0; i < bytes + 32; i++) diff += actual[i] != reference[i];
  for (size_t i = 0; i < 16; i++) diff += actual[i] != 0xcd || actual[bytes + 16 + i] != 0xcd;
  if (rr != 0) for (size_t i = 0; i < bytes; i++) diff += actual[16 + i] != 0xcd;
  if (rr == 0) for (size_t i = 0; i < M * N; i++) {
    uint16_t v = ((uint16_t *)(reference + 16))[i];
    sub += (v & 0x7f80) == 0 && (v & 0x7f); nz += v == 0x8000;
  }
  int ok = rr == vr && !diff && vlenb * 8 == (unsigned)atoi(argv[3]);
  printf("%s M=%zu K=%zu N=%zu mode=%d seed=%s rc=%d,%d diff=%zu subnormal_out=%zu negzero_out=%zu instr_merlin=%lu instr_generated=%lu vlen=%lu\n",
    ok ? "PASS" : "FAIL", M,K,N,mode,argv[2],rr,vr,diff,sub,nz,t1-t0,t2-t1,vlenb*8);
  free(a); free(w); free(wt); free(reference); free(actual); free(z);
  return !ok;
}
