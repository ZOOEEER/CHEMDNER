# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 09:09:38 2020

@author: hanyl
"""

class a:
    def __init__(self):
        self.attrs = {'a':1,'b':2}
        self.oo()
    
    def __getattr__(self, name):
        cls = type(self)
        if name in self.attrs.keys():
             return self.attrs[name]
        msg = '{.__name__!r} object has no attribute {!r}'
        raise AttributeError(msg.format(cls, name))
    
    def setpara(self, name, value):
        if not name in self.__dict__.keys():    
            self.attrs[name] = value
        else:
            print('Naming conflict.')
    def oo(self):
        print(self.a)
    
    def ii(self, **kwargs):
        print(kwargs)
    
def func():
    pass

f = func
