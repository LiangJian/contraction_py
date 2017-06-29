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
        alphabeta=string.ascii_lowercase
        nickname1 = alphabeta[:len(self.N)]
        nickname1_dict = {i:j for i,j in zip(self.N,list(nickname1))}

        nickname2 = ''#TODO
        nickname2_dict = {}
        for s in range(len(y_.N)):
            if y_.N[s] in self.N:
                nickname2 += nickname1_dict[y_.N[s]]
                nickname2_dict[y_.N[s]] = nickname1_dict[self.N[s]]
            else:
                nickname2 += list(string.ascii_lowercase)[count]
                nickname2_dict[y_.N[s]] = list(string.ascii_lowercase)[count]
                count += 1

        same_name_list = [s for s in self.N if s in y_.N]
        new_name_list = self.N + [s for s in y_.N if s not in same_name_list]

        self.rule = nickname1 + ',' + nickname2

        same = [s for s in y_.N if s in self.N]

        left2 =[s for s in y_.N if s not in same]

        left1 =[s for s in self.N if s not in same]

        new_name = [ s for s in left1 +  left2 ]

        if result_str_ != tuple(''):
            self.rule += '->'
            for s in new_name: #TODO
                if s not in result_str_:
                    print('wrong rule...')
                    exit(-1)
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
            if conjugate_:
                return self.rule, Tensor(name_=result_str_, t_=np.einsum(self.rule, self.T, y_.T.conjugate()))
            else:
                return self.rule, Tensor(name_=result_str_, t_=np.einsum(self.rule, self.T, y_.T))



        for s in left1:
            self.rule += nickname1_dict[left1[s]]
        for s in left2:
            self.rule += nickname2_dict[left2[s]]

        if conjugate_:
            return self.rule, Tensor(name_=tuple(new_name), t_=np.einsum(self.rule, self.T, y_.T.conjugate()))
        else:
            return self.rule, Tensor(name_=tuple(new_name), t_=np.einsum(self.rule, self.T, y_.T))
