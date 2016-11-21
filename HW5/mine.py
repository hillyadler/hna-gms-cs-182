import sys
sys.dont_write_bytecode = True
from hw5 import TextClassifier

cut = TextClassifier()

def test(o, e, val = 1, tol=.01):
    if type(o) is list:
        err = 0
        for i in xrange(len(o)):
            err += float(abs(o[i]-e[i])) / (e[i] + .000000000000000001)
        tol *= len(o)
    else:
        err = float(abs(o-e)) / (e + .000000000000000001)
    if err < tol:
        global points
        points += val
        return str(val) + "/" + str(val)
    else:
        return str(0) + "/" + str(val)

points = 0

print "#### MINI ####"
cut.q4('mini.train')
print "Right number of words:", len(cut.dict) == 8, "and ratings:", sum(cut.nrated) == 5
cut.q5()
print "q5 generates right F:", test(sum(cut.F, []), [2.01, 2.01, 2.71, 2.71, 2.01, 2.71, 2.01, 1.32, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 1.5, 1.95, 1.95, 2.64, 2.64, 1.95, 1.95, 1.95, 1.95, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 1.5, 1.79, 1.79, 1.79, 1.79, 2.48, 2.48, 2.48, 2.48])
ans = cut.q6('mini.valid')
print "q6 has right predictions:", test(ans[0], [4, 0, 2], 1)
print "q6 has right accuracy:", test(ans[1], 1./3, 1)
alpha = cut.q7('mini.valid')
cut.q5(alpha)
ans = cut.q6('mini.valid')
print "q7 produces better accuracy:", test(int(ans[1] >= 1./3 and alpha != 1), 1, 2)
cut.q5()
ans = cut.q8()
print "q8 hallucinates right words:", test(int(ans[2][0] == 'not' and ans[4][2] in ['182', 'cs'] and set(['rocks', '!']) == set(ans[4][:2]) and ans[0][0] == '.'), 1, 2)
