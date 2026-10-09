// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (C) 2026 Rhizomatica
 * Author: Rafael Diniz <rafael@rhizomatica.org>
 *
 * Based on Bruce Perens' ham_digital_modes
 * (https://github.com/BrucePerens/hams_open).
 */

/**
 * @file
 * @brief Encoder-side DTMF and single-tone detection.
 *
 * Ported from ham_digital_modes' float tone_detect.rs, whose behaviour was
 * calibrated against the DVSI AMBE-3000 chip: DTMF digits are detected in
 * their first frame, and single tones from 400 Hz to 3800 Hz and at 200 Hz,
 * with index round(f / 31.25 Hz). Call-progress tone pairs (dial, ring,
 * busy and 350 + 490 Hz), which DVSI's D-STAR encoder also sends as tone
 * frames, are detected by a joint fit of each pair.
 */

#include "mbe_tone_detect.h"

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const double mbe_dtmf_row_hz[4] = {697.0, 770.0, 852.0, 941.0};
static const double mbe_dtmf_col_hz[4] = {1209.0, 1336.0, 1477.0, 1633.0};

/* Fraction of a frame's energy one sinusoid must explain for the frame to be
 * classified as a single tone. A threshold of 0.9 classified voiced speech
 * dominated by its fundamental as a tone. */
#define MBE_TONE_SINGLE_MIN_EXPLAINED 0.99
/* Explained-energy fraction required of a 200 Hz tone, the only tone below
 * 400 Hz that is reported. */
#define MBE_TONE_LOW_MIN_EXPLAINED    0.999

/* Amplitude of the least-squares sinusoid of frequency hz in x. */
static double
mbe_tone_fit_sinusoid(const double* x, double hz) {
    double w = 2.0 * M_PI * hz / 8000.0;
    double sc = 0.0;
    double cc = 0.0;
    double sn = 0.0;
    double xc = 0.0;
    double xs = 0.0;
    for (int i = 0; i < MBE_TONE_DETECT_FRAME; i++) {
        double s = sin(w * (double)i);
        double c = cos(w * (double)i);
        cc += c * c;
        sn += s * s;
        sc += s * c;
        xc += x[i] * c;
        xs += x[i] * s;
    }
    double det = (cc * sn) - (sc * sc);
    if (fabs(det) < 1e-9) {
        return 0.0;
    }
    double a = ((xc * sn) - (xs * sc)) / det;
    double b = ((xs * cc) - (xc * sc)) / det;
    return sqrt((a * a) + (b * b));
}

/* Fraction of the frame's energy explained by independently fitted sinusoids. */
static double
mbe_tone_explained_fraction(const double* x, const double* hzs, int count) {
    double total = 0.0;
    double explained = 0.0;
    for (int i = 0; i < MBE_TONE_DETECT_FRAME; i++) {
        total += x[i] * x[i];
    }
    for (int i = 0; i < count; i++) {
        double a = mbe_tone_fit_sinusoid(x, hzs[i]);
        explained += a * a / 2.0 * (double)MBE_TONE_DETECT_FRAME;
    }
    if (total > 0.0) {
        return fmin(explained / total, 1.5);
    }
    return 0.0;
}

/* Solve the 4x4 system g[][0..3] c = g[][4] by Gaussian elimination with
 * partial pivoting; returns 0 if the system is singular. */
static int
mbe_tone_solve4(double g[4][5], double c[4]) {
    for (int col = 0; col < 4; col++) {
        int pivot = col;
        for (int r = col + 1; r < 4; r++) {
            if (fabs(g[r][col]) > fabs(g[pivot][col])) {
                pivot = r;
            }
        }
        if (fabs(g[pivot][col]) < 1e-9) {
            return 0;
        }
        for (int k = 0; k < 5; k++) {
            double t = g[col][k];
            g[col][k] = g[pivot][k];
            g[pivot][k] = t;
        }
        for (int r = col + 1; r < 4; r++) {
            double factor = g[r][col] / g[col][col];
            for (int k = col; k < 5; k++) {
                g[r][k] -= factor * g[col][k];
            }
        }
    }
    for (int r = 3; r >= 0; r--) {
        double sum = g[r][4];
        for (int k = r + 1; k < 4; k++) {
            sum -= g[r][k] * c[k];
        }
        c[r] = sum / g[r][r];
    }
    return 1;
}

/*
 * Joint least-squares fit of two sinusoids at f1 and f2 Hz: the projection of
 * x onto the span of their four cosine and sine components. Fitting the pair
 * jointly keeps closely spaced tones (440 and 480 Hz are only two cycles apart
 * over one frame) from being counted twice. Returns the fraction of the
 * frame's energy explained, and each tone's amplitude.
 */
static double
mbe_tone_fit_pair(const double* x, double f1, double f2, double* a1, double* a2) {
    double basis[4][MBE_TONE_DETECT_FRAME];
    double g[4][5];
    double rhs[4];
    double c[4];
    double total = 0.0;
    double w1 = 2.0 * M_PI * f1 / 8000.0;
    double w2 = 2.0 * M_PI * f2 / 8000.0;

    *a1 = 0.0;
    *a2 = 0.0;
    for (int i = 0; i < MBE_TONE_DETECT_FRAME; i++) {
        basis[0][i] = cos(w1 * (double)i);
        basis[1][i] = sin(w1 * (double)i);
        basis[2][i] = cos(w2 * (double)i);
        basis[3][i] = sin(w2 * (double)i);
        total += x[i] * x[i];
    }
    /* Normal equations G c = B^T x, with B^T x as the fifth column. */
    for (int r = 0; r < 4; r++) {
        for (int k = 0; k < 4; k++) {
            double sum = 0.0;
            for (int i = 0; i < MBE_TONE_DETECT_FRAME; i++) {
                sum += basis[r][i] * basis[k][i];
            }
            g[r][k] = sum;
        }
        rhs[r] = 0.0;
        for (int i = 0; i < MBE_TONE_DETECT_FRAME; i++) {
            rhs[r] += basis[r][i] * x[i];
        }
        g[r][4] = rhs[r];
    }
    if (!(total > 0.0) || !mbe_tone_solve4(g, c)) {
        return 0.0;
    }
    /* The energy of the projection is (B^T x) . c. */
    double explained = 0.0;
    for (int r = 0; r < 4; r++) {
        explained += rhs[r] * c[r];
    }
    *a1 = sqrt((c[0] * c[0]) + (c[1] * c[1]));
    *a2 = sqrt((c[2] * c[2]) + (c[3] * c[3]));
    return explained / total;
}

/* Index of the largest value; ties resolve to the highest index. */
static int
mbe_tone_arg_max(const double v[4]) {
    int best = 0;
    for (int i = 1; i < 4; i++) {
        if (v[i] >= v[best]) {
            best = i;
        }
    }
    return best;
}

/* DTMF: the strongest row and column, together explaining the frame. */
static int
mbe_tone_detect_dtmf(const double* frame, struct mbe_tone_detection* out) {
    double row_amps[4];
    double col_amps[4];
    for (int i = 0; i < 4; i++) {
        row_amps[i] = mbe_tone_fit_sinusoid(frame, mbe_dtmf_row_hz[i]);
        col_amps[i] = mbe_tone_fit_sinusoid(frame, mbe_dtmf_col_hz[i]);
    }
    int r = mbe_tone_arg_max(row_amps);
    int c = mbe_tone_arg_max(col_amps);
    double ra = row_amps[r];
    double ca = col_amps[c];
    if (!(ra > 100.0 && ca > 100.0 && fmax(ra / ca, ca / ra) < 3.0)) {
        return 0;
    }
    const double hzs[2] = {mbe_dtmf_row_hz[r], mbe_dtmf_col_hz[c]};
    if (!(mbe_tone_explained_fraction(frame, hzs, 2) > 0.85)) {
        return 0;
    }
    out->kind = MBE_TONE_DETECT_DTMF;
    out->row = r;
    out->col = c;
    out->amplitude = (ra + ca) / 2.0;
    return 1;
}

/* Call-progress tone pairs, in the order of TIA-102.BABA-1 Table 9 tone
 * indices 160-163 (D-STAR indices 144-147). */
static const double mbe_call_progress_hz[4][2] = {{350.0, 440.0}, {440.0, 480.0}, {480.0, 620.0}, {350.0, 490.0}};

/* Fraction of a frame's energy a call-progress pair must explain. Its tones
 * lie in the range of voiced speech harmonics, so the threshold is stricter
 * than DTMF's. */
#define MBE_TONE_CALL_PROGRESS_MIN_EXPLAINED 0.95

/* Explained-energy fraction of call-progress pair k with its tones shifted by
 * the relative offsets s1 and s2. */
static double
mbe_tone_fit_shifted_pair(const double* frame, int k, double s1, double s2) {
    double a1;
    double a2;
    return mbe_tone_fit_pair(frame, mbe_call_progress_hz[k][0] * (1.0 + s1), mbe_call_progress_hz[k][1] * (1.0 + s2),
                             &a1, &a2);
}

/*
 * Whether pair k's tones are within 1.25% of their nominal frequencies. Each
 * tone's offset is estimated as the shift of the best joint fit, first on a 1%
 * grid over +-3%, then on a 0.25% grid within 0.75% of that point; for a clean
 * tone pair this estimate is within 0.25% of the true offset at every frame
 * position. DVSI's encoder rejects a ring tone at 433 + 484 Hz (440 Hz 1.6%
 * low).
 */
static int
mbe_tone_call_progress_on_frequency(const double* frame, int k) {
    double best = -1.0;
    double s1 = 0.0;
    double s2 = 0.0;
    for (int i = -3; i <= 3; i++) {
        for (int j = -3; j <= 3; j++) {
            double fraction = mbe_tone_fit_shifted_pair(frame, k, 0.01 * (double)i, 0.01 * (double)j);
            if (fraction > best) {
                best = fraction;
                s1 = 0.01 * (double)i;
                s2 = 0.01 * (double)j;
            }
        }
    }
    double c1 = s1;
    double c2 = s2;
    for (int i = -3; i <= 3; i++) {
        for (int j = -3; j <= 3; j++) {
            double t1 = c1 + (0.0025 * (double)i);
            double t2 = c2 + (0.0025 * (double)j);
            double fraction = mbe_tone_fit_shifted_pair(frame, k, t1, t2);
            if (fraction > best) {
                best = fraction;
                s1 = t1;
                s2 = t2;
            }
        }
    }
    return fabs(s1) <= 0.0126 && fabs(s2) <= 0.0126;
}

/* Call progress: the pair best explaining the frame, with comparable levels
 * and frequencies within tolerance. */
static int
mbe_tone_detect_call_progress(const double* frame, struct mbe_tone_detection* out) {
    int best = -1;
    double best_fraction = 0.0;
    double best_amplitude = 0.0;
    for (int k = 0; k < 4; k++) {
        double a1;
        double a2;
        double fraction = mbe_tone_fit_pair(frame, mbe_call_progress_hz[k][0], mbe_call_progress_hz[k][1], &a1, &a2);
        if (!(a1 > 100.0 && a2 > 100.0 && fmax(a1 / a2, a2 / a1) < 3.0)) {
            continue;
        }
        if (fraction > MBE_TONE_CALL_PROGRESS_MIN_EXPLAINED && fraction > best_fraction) {
            best = k;
            best_fraction = fraction;
            best_amplitude = (a1 + a2) / 2.0;
        }
    }
    if (best < 0 || !mbe_tone_call_progress_on_frequency(frame, best)) {
        return 0;
    }
    out->kind = MBE_TONE_DETECT_CALL_PROGRESS;
    out->index = best;
    out->amplitude = best_amplitude;
    return 1;
}

/* Single tone: coarse scan for the strongest sinusoid, then refine. */
static int
mbe_tone_detect_single(const double* frame, struct mbe_tone_detection* out) {
    double best_amp = 0.0;
    double best_hz = 0.0;
    /* 150 to 3900 Hz in 12.5 Hz steps, then +-12.5 Hz in 1 Hz steps. Both
     * steps are exactly representable in binary floating point, so these
     * values equal those of the reference's accumulated loop variables. */
    for (int step = 0; step <= 300; step++) {
        double hz = 150.0 + (12.5 * (double)step);
        double a = mbe_tone_fit_sinusoid(frame, hz);
        if (a > best_amp) {
            best_amp = a;
            best_hz = hz;
        }
    }
    double amp = best_amp;
    double hz = best_hz;
    for (int step = 0; step <= 25; step++) {
        double f = best_hz - 12.5 + (double)step;
        double a = mbe_tone_fit_sinusoid(frame, f);
        if (a > amp) {
            amp = a;
            hz = f;
        }
    }
    double frac = mbe_tone_explained_fraction(frame, &hz, 1);
    if (!(amp > 100.0 && frac > MBE_TONE_SINGLE_MIN_EXPLAINED)) {
        return 0;
    }
    int index = (int)round(hz / 31.25);
    /* Below 400 Hz, voiced speech is itself close to a pure sinusoid, so only a
     * component at 200 Hz that explains nearly all of the frame's energy is
     * accepted there. */
    int in_range =
        (index >= 13 && index <= 122) || (index == 6 && fabs(hz - 200.0) < 3.0 && frac > MBE_TONE_LOW_MIN_EXPLAINED);
    if (!in_range) {
        return 0;
    }
    out->kind = MBE_TONE_DETECT_SINGLE;
    out->index = index;
    out->hz = hz;
    out->amplitude = amp;
    return 1;
}

enum mbe_tone_detect_kind
mbe_tone_detect(const double frame[MBE_TONE_DETECT_FRAME], struct mbe_tone_detection* out) {
    double energy = 0.0;
    memset(out, 0, sizeof(*out));

    for (int i = 0; i < MBE_TONE_DETECT_FRAME; i++) {
        energy += frame[i] * frame[i];
    }
    energy /= (double)MBE_TONE_DETECT_FRAME;
    if (energy < 100.0 * 100.0 / 2.0) {
        return MBE_TONE_DETECT_NONE; /* below the mean power of a sinusoid of amplitude 100 */
    }
    if (mbe_tone_detect_dtmf(frame, out) || mbe_tone_detect_call_progress(frame, out)
        || mbe_tone_detect_single(frame, out)) {
        return out->kind;
    }
    return MBE_TONE_DETECT_NONE;
}
