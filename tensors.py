import numpy as np
import string


class Tensor:

    def __init__(self, shape_=0, name_=tuple(''), t_=None):
        if t_ is None:
            self.T = np.zeros(shape=shape_, dtype='complex128')

        else:
            self.T = t_
        self.N = name_
        self.rule = ''
        if self.T.ndim != len(self.N):
            print('wrong size of initialization.')
            exit(-1)

    def contract(self, y_, conjugate_=False, result_str_=tuple('')):
        count = 0
        nickname1 = ''
        nickname1_dict = {}
        for s in self.N:
            nickname1 += list(string.ascii_lowercase)[count]
            nickname1_dict[s] = list(string.ascii_lowercase)[count]
            count += 1
        nickname2 = ''
        nickname2_dict = {}
        for s in range(len(y_.N)):
            if y_.N[s] in self.N:
                nickname2 += nickname1_dict[y_.N[s]]
                nickname2_dict[y_.N[s]] = nickname1_dict[self.N[s]]
            else:
                nickname2 += list(string.ascii_lowercase)[count]
                nickname2_dict[y_.N[s]] = list(string.ascii_lowercase)[count]
                count += 1

        self.rule = nickname1
        self.rule += ','
        self.rule += nickname2
        self.rule += '->'
        if result_str_ != tuple(''):
            nickname3 = ''
            for s in result_str_:
                if s in self.N:
                    nickname3 += nickname1_dict[s]
                else:
                    if s in y_.N:
                        nickname3 += nickname2_dict[s]
                    else:
                        print('wong rule...')
                        exit(-1)
            self.rule += nickname3
            return self.rule, Tensor(name_=result_str_, t_=np.einsum(self.rule, self.T, y_.T.conjugate()))

        count = 0
        same = ''
        for s2 in nickname2:
            if s2 in nickname1:
                count += 1
                same += s2

        left2 = ''
        for s2 in nickname2:
            if s2 not in same:
                left2 += s2

        left1 = ''
        for s1 in nickname1:
            if s1 not in same:
                left1 += s1

        self.rule += left1 + left2
        new_name = []
        for s in left1:
            new_name.append(s)
        for s in left2:
            new_name.append(s)
        if conjugate_:
            return self.rule, Tensor(name_=tuple(new_name), t_=np.einsum(self.rule, self.T, y_.T))
        else:
            return self.rule, Tensor(name_=tuple(new_name), t_=np.einsum(self.rule, self.T, y_.T.conjugate()))
