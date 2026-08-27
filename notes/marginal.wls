#!/usr/bin/env wolframscript
(* Does the Week 2 marginal have a closed form?
   usage: wolframscript -file marginal.wls *)

dp = 2; Sp = 1; u = 1/2; Su = 1/10;

expo[g_] := -(d - dp)^2/(2 Sp) - (u - g)^2/(2 Su);

Print["── exponent powers, the argument the lesson actually makes ──"];
Print["linear   g=d      : ", Union[Exponent[Expand[expo[d]], d, List]]];
Print["inv-sq   g=1/d^2  : ", Union[Exponent[Expand[expo[1/d^2]], d, List]]];

Print[""];
Print["── linear case, for contrast ──"];
lin = TimeConstrained[Integrate[Exp[expo[d]], {d, -Infinity, Infinity}], 300, "TIMEOUT"];
Print["Integrate = ", lin];
Print["N[40]     = ", If[lin === "TIMEOUT", "-", N[lin, 40]]];

Print[""];
Print["── the nonlinear case: does Integrate return a closed form? ──"];
nl = TimeConstrained[Integrate[Exp[expo[1/d^2]], {d, 0, Infinity}], 900, "TIMEOUT"];
Print["Head      = ", Head[nl]];
Print["FreeQ[Integrate] (i.e. fully evaluated?) = ", FreeQ[nl, Integrate]];
Print["result    = ", Short[nl, 6]];

Print[""];
Print["── indefinite antiderivative? ──"];
anti = TimeConstrained[Integrate[Exp[expo[1/d^2]], d], 600, "TIMEOUT"];
Print["Head = ", Head[anti], "   free of Integrate? ", FreeQ[anti, Integrate]];
Print[Short[anti, 4]];

Print[""];
Print["── high-precision numeric value of the marginal (up to the 1/Sqrt[2 Pi ...] factors) ──"];
nv = TimeConstrained[
  NIntegrate[Exp[expo[1/d^2]], {d, 0, Infinity}, WorkingPrecision -> 30,
             MaxRecursion -> 30], 600, "TIMEOUT"];
Print["NIntegrate (30 digits) = ", nv];
Print["Precision              = ", If[NumberQ[nv], Precision[nv], "-"]];

If[NumberQ[nv],
  Print[""];
  Print["── is that constant recognisable? (PSLQ against a small basis) ──"];
  basis = {nv, 1, Pi, Sqrt[Pi], E, Sqrt[2], Log[2]};
  Print["FindIntegerNullVector = ",
        TimeConstrained[FindIntegerNullVector[N[basis, 28]], 120, "TIMEOUT"]]];

Exit[0]
