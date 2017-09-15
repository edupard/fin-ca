def test():

    a = 1

    def b():
        nonlocal a
        a += 1

    b()
    print(a)

test()

