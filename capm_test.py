from portfolio.capm import Capm

exp = [0.05, -0.03]
cov = [[0.05*0.05,0.05*0.03],[0.05*0.03, 0.03 * 0.03]]
print(cov)

capm = Capm(2)
capm.init()
i = 0
while i<10000:
    w, sharpe, constraint = capm.get_params(exp, cov)
    if w is None:
        break
    print("Iteration: %d Sharpe: %.2f Constraint: %.6f" % (i, sharpe, constraint))
    print(w)
    capm.fit(exp, cov)
    capm.rescale_weights()
    i += 1