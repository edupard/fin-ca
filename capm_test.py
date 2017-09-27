from portfolio.capm import Capm

exp = [0.05, -0.03]
cov = [[0.05*0.05,0.05*0.03],[0.05*0.03, 0.03 * 0.03]]
print(cov)

capm = Capm(2)
capm.init()
lambda_coef = 1.0
max_constraint = 0.0
while True:
    w, sharpe, constraint = capm.fit(exp, cov, lambda_coef)
    print("Sharpe: %.2f Constraint: %.6f Lambda: %.6f" % (sharpe, constraint, lambda_coef))
    print(w)
    # if constraint > max_constraint:
    #     lambda_coef += constraint - max_constraint
    #     max_constraint = constraint
        # capm.reset_optimizer()

    if constraint > 0.01:
        lambda_coef = lambda_coef + 1
        # lambda_coef = lambda_coef * 1
    #     capm.reset_optimizer()