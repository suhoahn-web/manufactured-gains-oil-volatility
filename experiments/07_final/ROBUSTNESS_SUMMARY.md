# Robustness programme: what each check was for and what it returned

Ten checks, run after the manuscript was drafted, each aimed at an objection a
referee can raise. Two required retraining; the rest reuse the per-week pooled
losses. Everything below is reproducible from the scripts named in each row.

| # | Objection it answers | Verdict | Effect on the paper |
|---|---|---|---|
| R1 | MAE is not robust to proxy noise, so the reversal may be an artefact | **Defused** | Reversal holds under MSE, which is robust |
| R1b | The reversal was called significant on unadjusted p-values | **Confirmed as a flaw** | Section 5.3 must be reworded |
| R2 | The 2020 decomposition is shown only under one loss | **Strengthened** | Holds under all three; deterioration is significant under both symmetric losses |
| R3 | The negative-price weeks contaminate exactly the decisive year | **Survives, with a caveat** | Conclusion unchanged; the h = 12 magnitude falls by three quarters |
| R4 | 156 test weeks were reused from hyperparameter selection | **Survives** | But raises a new comparison a referee would otherwise find first |
| R4b | On the clean subsample the model appears to beat the benchmark | **Explained** | It is a 2020-share effect; the doubly-clean sample restores the null |
| R5 | The design cannot detect the gains this literature reports | **Reframed** | 56 to 219 years of weekly data would be needed |
| S1 | The auxiliary-task contrast is confounded by the stopping rule | **Defused, with a mechanism** | The confound is empirically nil and it is clear why |
| S2 | The constraint was never binding, so the null is about its strength | **Resolved into a trade-off** | At binding weights the constraint significantly harms accuracy |
| S3 | Anchoring removes the channel through which fundamentals act | **Defused, and it replicates the mechanism** | The null holds without the anchor, and the manufactured gain reappears there |

Scripts: `referee_robustness.py` (R1–R5), `extra_configs_inference.py` (S1–S2),
`../06_robustness/expanding_window_extra.py` and
`../06_robustness/expanding_window_unanchored.py` (the two retraining runs).

---

## The two findings that change what the paper claims

### 1. Excluding 2020, the specifications are significantly worse — not merely worse

The manuscript reports this contrast under the quasi-likelihood loss only,
where Holm-adjusted p-values sit between 0.058 and 0.066. Under both symmetric
losses the same contrast clears 5%:

| M5, λ_B = 0.5, excluding 2020 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|
| QLIKE | +14.5% (0.093) | +19.6% (0.093) | +28.7% (0.079) | +16.1% (0.298) |
| MAE | +11.6% (**0.023**) | +16.4% (**0.022**) | +21.6% (**0.022**) | +20.7% (**0.023**) |
| MSE | +22.4% (**0.033**) | +28.4% (**0.033**) | +41.2% (**0.033**) | +32.1% (**0.037**) |

Two consequences. The claim in Section 5.2 can be raised from a sign reversal
to a significant deterioration. And the power objection weakens: the design
does detect differences at conventional levels, in the direction opposite to
the hypothesis.

### 2. The constraint does bind at higher weights, and then it hurts

The sweep in the manuscript stops at 0.5, where the balance-consistency error
moves by 2.3% — less than the dispersion across seeds — which leaves open
whether the penalty was ever binding. At weights of 2 and 5 it plainly is, and
the accuracy cost is significant.

| Constraint weight | Balance-consistency error (10³ bbl/week) | versus no penalty |
|---|---|---|
| 0 | 5,223 | — |
| 2.0 | 2,467 | −52.8% |
| 5.0 | 2,349 | −55.0% |

| Versus the unpenalised network, Holm-adjusted p | h = 1 | h = 2 | h = 4 | h = 8 |
|---|---|---|---|---|
| λ_B = 2.0, MAE | 3.21 (**0.004**) | 3.55 (**0.002**) | 3.57 (**0.002**) | 3.03 (**0.005**) |
| λ_B = 5.0, MAE | 3.75 (**0.001**) | 3.88 (**0.001**) | 3.66 (**0.001**) | 2.87 (**0.008**) |
| λ_B = 5.0, MSE | 3.21 (**0.007**) | 3.22 (**0.007**) | 2.77 (**0.018**) | 1.99 (0.093) |

This converts a gap into a result. There is no weight at which the constraint
both binds and helps: below 0.5 it does not bind, and at 2 and above it binds
and significantly degrades forecasts.

---

### 3. The stopping-rule confound has no effect, and the reason is informative

Section 4.5 records that M4 and M5 monitor the composite objective for early
stopping while M4arch monitors the forecast term alone, and warns that the
M4arch-to-M4 contrast therefore mixes the auxiliary gradient with the stopping
criterion. Re-running M4 with the forecast term monitored returns forecasts
identical to the reported ones — a maximum absolute difference of zero across
487 weeks, five horizons and three losses. The auxiliary-task result in Section
5.4 is therefore the auxiliary gradient alone.

Tracing both validation series for one window and seed shows why. The
auxiliary component is nearly flat across epochs (0.742 to 0.723) while the
forecast term moves an order of magnitude more (0.447 to 0.678 within eight
epochs), so the composite is close to an affine transformation of the forecast
term and the two monitors reach their minimum at the same epoch.

The trace carries a second fact worth reporting. The validation minimum falls
at epoch 2. With 559 training rows and roughly 26,000 parameters these networks
are at their best after three passes and overfit immediately afterwards, which
is the same over-parameterisation that the seed dispersion in Section 5.5
records from a different angle.

### 4. The anchor is not what produced the null, and dropping it reproduces the manufactured gain

The benchmark absorbs the persistence of realized volatility. If the storage
mechanism acts through persistence, anchoring would remove the channel through
which fundamentals could help, and the null would be an artefact of the design.
Re-estimating the price-only and physical specifications on the log realized
volatilities themselves settles it.

Without the anchor the physical block still does not significantly beat the
price-only specification: the point estimates now favour it under the
quasi-likelihood and squared losses at every horizon, but no Holm-adjusted
p-value falls below 0.686, and under absolute error the direction is mixed.
The reading that anchoring suppressed the effect is therefore unavailable.

The anchor's cost, which Section 4.2 asserts without a number, is +7.5% and
+8.4% of the benchmark's quasi-likelihood loss at one and two weeks for the
price-only specification. At eight and twelve weeks the unanchored version is
in fact slightly better, so the justification holds only in the horizon range
the section claims.

The more useful result is what the unanchored physical specification does on
the full sample. It beats the benchmark by 15.2% and 17.4% at eight and twelve
weeks under the quasi-likelihood loss — a larger apparent gain than anything in
the paper — with Holm-adjusted p-values of 1.000. Excluding 2020 it is worse
than the benchmark at every horizon. The manufactured gain is thus not a
property of the anchored ladder; it reappears in a specification family the
paper does not otherwise report, which is what a claim about reporting practice
rather than about one architecture requires.

## The check that a referee would otherwise have run first

Removing the 156 weeks that overlap the hyperparameter-selection span leaves
331 weeks on which the most heavily constrained specification has the lowest
loss under all three loss functions — by 23% under the quasi-likelihood loss at
twelve weeks. Read alone, that reverses the paper's conclusion.

It does not survive being crossed with the other subsample criterion. Removing
2017–2019 raises the share of 2020 weeks from 10.7% to 15.7%, and Section 5.2
establishes that those weeks carry the whole apparent advantage.

| Sample | n | 2020 share | M5 (λ_B = 0.5) versus HAR, QLIKE, h = 12 |
|---|---|---|---|
| Full pooled | 487 | 10.7% | −13.7% (p = 1.000) |
| Excluding the selection overlap | 331 | 15.7% | −23.0% (p = 1.000) |
| Excluding 2020 | 435 | 0% | +16.1% (p = 0.298) |
| **Excluding both** | **279** | **0%** | **+3.8% (p = 0.681)** |

On the doubly-clean sample every specification is worse than the benchmark
under all three losses at essentially every horizon. The conclusion holds, but
the four-row table belongs in the paper: presented only after a referee finds
the second row, the explanation reads as a rescue.

---

## The caveat that has to be stated

Dropping every origin whose target window spans 20 April 2020 leaves the 2020
decomposition qualitatively intact — the sign still reverses — but a large part
of the long-horizon magnitude was carried by those origins.

| M5, λ_B = 0.1, QLIKE difference against HAR | h = 1 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|
| 2020, as reported | −0.0567 | −0.0767 | −0.1124 | −0.1283 |
| 2020, contaminated origins removed | −0.0523 | −0.0681 | −0.0635 | −0.0338 |
| Excluding 2020 | +0.0043 | +0.0099 | +0.0131 | +0.0110 |

At twelve weeks roughly three quarters of the 2020 advantage disappears with
twelve origins. Section 3.1 already flags that the affected windows fall inside
the decisive episode; the quantification belongs beside it.
