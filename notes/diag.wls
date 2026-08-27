#!/usr/bin/env wolframscript
(* Verify the diagnosis of Jonathan's active-inference notebook symbolically.
   Model: s1 ~ Exp(rate a), s2 ~ Exp(rate b) under the mean-field q.
   usage: wolframscript -file diag.wls *)

fails = 0; checks = 0;
gate[l_, ok_] := (checks++;
  Which[ok === True, Print["PASS  ", l],
        ok === False, Print["FAIL  ", l]; fails++,
        True, Print["ABORT non-boolean: ", l]; Exit[2]]);

d1 = ExponentialDistribution[a]; d2 = ExponentialDistribution[b];
ex[f_] := Expectation[Expectation[f, s1 \[Distributed] d1], s2 \[Distributed] d2];

Print["── raw moments of an exponential ──"];
Print["E[s1^k], k=1..4 : ", Table[Expectation[s1^k, s1 \[Distributed] d1], {k, 4}]];

Print[""];
Print["── the two squared-error terms in the notebook's free energy ──"];
t1 = FullSimplify[ex[(oy - s2 s1^2)^2], Assumptions -> a > 0 && b > 0];
t2 = FullSimplify[ex[(oi - s2^2 s1)^2], Assumptions -> a > 0 && b > 0];
Print["E[(oy - s2 s1^2)^2] = ", t1];
Print["E[(oi - s2^2 s1)^2] = ", t2];
gate["term 1 matches oy^2 - 4 oy/(a^2 b) + 48/(a^4 b^2)",
     Simplify[t1 - (oy^2 - 4 oy/(a^2 b) + 48/(a^4 b^2))] === 0];
gate["term 2 matches oi^2 - 4 oi/(a b^2) + 48/(a^2 b^4)",
     Simplify[t2 - (oi^2 - 4 oi/(a b^2) + 48/(a^2 b^4))] === 0];

Print[""];
Print["── where the 48 comes from, and the factor of twelve ──"];
sec = FullSimplify[ex[(s2 s1^2)^2], Assumptions -> a > 0 && b > 0];
mn  = FullSimplify[ex[s2 s1^2], Assumptions -> a > 0 && b > 0];
Print["E[(s2 s1^2)^2] = ", sec, "     (E[s2 s1^2])^2 = ", Simplify[mn^2]];
gate["the second moment is 12x the square of the mean",
     Simplify[sec - 12 mn^2] === 0];
gate["so the variance of the predicted observation is 11x its squared mean",
     Simplify[(sec - mn^2) - 11 mn^2] === 0];

Print[""];
Print["── the family's rigidity: coefficient of variation ──"];
cv = FullSimplify[Sqrt[Variance[d1]]/Mean[d1], Assumptions -> a > 0];
Print["CV of Exp(a) = ", cv, "   (independent of the rate)"];
gate["an exponential q is locked at CV = 1", cv === 1];

Print[""];
Print["── the M-step: averaging rates versus averaging means ──"];
Print["Correct update  lambda = 1/Mean[{1/t1r, 1/t2r}]  is the HARMONIC mean of rates."];
Print["Notebook update lambda = Mean[{t1r, t2r}]        is the ARITHMETIC mean."];
am = (t1r + t2r)/2; hm = 2/(1/t1r + 1/t2r);
gate["arithmetic >= harmonic for positive rates, so the M-step biases the rate UP",
     Reduce[ForAll[{t1r, t2r}, t1r > 0 && t2r > 0, am >= hm], Reals] === True];
Print["ratio AM/HM at rates {0.9, 3.0} = ", N[(am/hm) /. {t1r -> 0.9, t2r -> 3.0}]];

Print[""];
If[fails > 0 || checks < 6,
  Print["checks run: ", checks, "   failures: ", fails]; Exit[1],
  Print["ALL ", checks, " GATES PASS"]];
