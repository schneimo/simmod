
class Test():

    def __init__(self):
        self.x = 5
        

t = Test()
print(t.x)
setattr(t, 'x', 6)
print(t.x)