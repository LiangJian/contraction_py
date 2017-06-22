import numpy as np
import os.path

def mass_eff_cosh(src,index_t):
    return np.arccosh((np.roll(src,+1,index_t)+np.roll(src,-1,index_t))/(2.*src))


def mass_eff_log(src,index_t):
    return np.log(np.roll(src,+1,index_t)/src)


def get_std_error(src,index_conf):
    return np.std(src,index_conf)/np.sqrt(src.shape[index_conf])


def do_jack(src,index_conf):
    tmp=np.sum(src,index_conf,keepdims=True)
    return (tmp-src)/(src.shape[index_conf]-1)


def get_jack_error(src,index_conf):
    return np.std(src,index_conf)/np.sqrt(src.shape[index_conf])*(src.shape[index_conf]-1.)


def combine(a_, b_, index_name_=''):
    index = a_.find_name(index_name_)
    # check heads of a and b
    tmp = np.concatenate((a_, b_), axis=index)
    tmp = Var(data=tmp, init_method='array data')
    tmp.head_data = a_.head_data
    tmp.head_data_['head']['one_dim']['n_indices'][index] = a_.shape[index] + b_.shape[index]
    for i in range(b_.shape[index]):
        tmp.head_data_['head']['one_dim']['indices'][index][a_.shape[index]+i] =\
         b_.head_data_['head']['one_dim']['indices'][index][i]
    tmp.update_meta()
    print(a_.head_data)

class Var(np.ndarray):

    def __new__(
        cls,
        name                    = "",
        tree                    = None, # tree object
        eventNumber             = None,
        eventWeight             = None,
        numberOfBins            = None, # binning
        binningLogicSystem      = None, # binning
        shape                   = None,
        filename                = '',
        scatter_file_name       = None,
        scatter_num             = None,
        scatter_index_name      = None,
        endian                  = '<',
        data                    = None,
        init_method             = 'shape'
        ):
        HeadType = np.dtype(
            [('head',
              [('n_dims', 'i4'),('one_dim',
                                 [('type', 'i4'),('n_indices', 'i4'),('indices', 'i4', 1024)
                                  ], 16)
               ])
             ]
        )
        typename = ["other", "x", "y", "z", "t", "d", "c", "d2",
                    "c2", "complex", "mass", "smear", "displacement",
                    "s_01", "s_02", "s_03", "s_11", "s_12", "s_13",
                    "d_01", "d_02", "d_03", "d_11", "d_12",
                    "d_13", "conf", "operator", "momentum", "direction",
                    "t2", "mass2", "column", "row",
                    "temporary", "temporary2", "temporary3", "temporary4",
                    "errorbar", "operator2", "param",
                    "fit_left", "fit_right", "jackknife", "jackknife2",
                    "jackknife3", "jackknife4", "summary",
                    "channel", "channel2", "eigen", "d_row", "d_col",
                    "c_row", "c_col", "parity", "noise",
                    "evenodd", "disp_x", "disp_y", "disp_z", "disp_t",
                    "t3", "t4", "t_source","t_current", "t_sink",
                    "bootstrap", "nothing"]

        init_methods = []
        # CHECK METHOD

        self = np.zeros(0).view(cls)
        if init_method == 'shape': # init from shape
            self = np.zeros(shape=shape).view(cls)
            self.head_data = np.zeros(1,dtype=HeadType)[0]
            self.head_data['head']['n_dims'] = len(shape)
            for i in range(len(shape)):
                self.head_data['head']['one_dim']['n_indices'][i] = shape[i]

            head_data_ = self.head_data
            n_dims_ = head_data_['head']['n_dims']
            type_ = head_data_['head']['one_dim']['type'][0:n_dims_]
            typename_ = [typename[i] for i in type_]
            n_indices_ = head_data_['head']['one_dim']['n_indices'][0:n_dims_]
            indices_ = head_data_['head']['one_dim']['indices'][0:n_dims_]
            self.indices = {}
            for i in range(n_dims_):
                self.indices[typename_[i]] = indices_[i][0:n_indices_[i]]
            self.type = typename_
            self.index = {}
            for i in range(n_dims_):
                self.index[typename_[i]] = i
            self.head_data = head_data_

        if init_method == 'array data': # init from list
            if data is not None:
                self = np.array(data).view(cls)
            else:
                print('wrong ini params')
                exit(-1)

        if init_method == 'scatter_iog_file':
            if scatter_file_name is None or scatter_num is None or scatter_index_name is None:
                print('wrong ini params')
                exit(-1)
            else:
                file_names_ = []
                for ic in range(scatter_num[0], scatter_num[1], scatter_num[2]):
                    if os.path.isfile(scatter_file_name % ic):
                        file_names_.append(scatter_file_name % ic)
                print(file_names_)

                tmp_ = Var(filename=file_names_[0], init_method="iog_file")
                shape_ = np.array(tmp_.shape)
                shape_[tmp_.find_name(name_=scatter_index_name)] = len(file_names_)
                self = np.zeros(shape=shape_).view(cls).view(cls)

                head_data_ = tmp_.head_data
                n_dims_ = head_data_['head']['n_dims']
                type_ = head_data_['head']['one_dim']['type'][0:n_dims_]
                typename_ = [typename[i] for i in type_]
                n_indices_ = head_data_['head']['one_dim']['n_indices'][0:n_dims_]
                indices_ = head_data_['head']['one_dim']['indices'][0:n_dims_]

                n_indices_[tmp_.find_name(name_=scatter_index_name)] = len(file_names_)

                for i in range(len(file_names_)):
                    tmp_ = Var(filename=file_names_[i], init_method="iog_file")
                    self[i, ...] = tmp_
                    indices_[tmp_.find_name(name_=scatter_index_name)][i] = tmp_.indices[scatter_index_name][0]

                self.indices = {}
                for i in range(n_dims_):
                    self.indices[typename_[i]] = indices_[i][0:n_indices_[i]]
                self.type = typename_
                self.index = {}
                for i in range(n_dims_):
                    self.index[typename_[i]] = i
                self.head_data = head_data_

        if init_method == 'scatter_Var_file':
            if scatter_file_name is None or scatter_num is None or scatter_index_name is None:
                print('wrong ini params')
                exit(-1)
            else:
                file_names_ = []
                for ic in range(scatter_num[0], scatter_num[1], scatter_num[2]):
                    if os.path.isfile(scatter_file_name % ic):
                        file_names_.append(scatter_file_name % ic)

                tmp_ = Var(name=file_names_[0],init_method="Var_file")
                shape_ = np.array(tmp_.shape)
                shape_[tmp_.find_name(name_=scatter_index_name)] = len(file_names_)
                self = np.zeros(shape=shape_).view(cls).view(cls)

                head_data_ = tmp_.head_data
                n_dims_ = head_data_['head']['n_dims']
                type_ = head_data_['head']['one_dim']['type'][0:n_dims_]
                typename_ = [typename[i] for i in type_]
                n_indices_ = head_data_['head']['one_dim']['n_indices'][0:n_dims_]
                indices_ = head_data_['head']['one_dim']['indices'][0:n_dims_]

                n_indices_[tmp_.find_name(name_=scatter_index_name)] = len(file_names_)

                for i in range(len(file_names_)):
                    tmp_ = Var(name=file_names_[i], init_method="Var_file")
                    self[i, ...] = tmp_
                    indices_[tmp_.find_name(name_=scatter_index_name)][i] = tmp_.indices[scatter_index_name][0]

                self.indices = {}
                for i in range(n_dims_):
                    self.indices[typename_[i]] = indices_[i][0:n_indices_[i]]
                self.type = typename_
                self.index = {}
                for i in range(n_dims_):
                    self.index[typename_[i]] = i
                self.head_data = head_data_

        if init_method == 'iog_file':
            if filename == '':
                print('wrong ini params')
                exit(-1)
            else:
                head_data_ = np.fromfile(filename, dtype=HeadType, count=1)[0]
                n_dims_ = head_data_['head']['n_dims']
                type_ = head_data_['head']['one_dim']['type'][0:n_dims_]
                typename_ = [typename[i] for i in type_]
                n_indices_ = tuple(head_data_['head']['one_dim']['n_indices'][0:n_dims_])
                indices_ = head_data_['head']['one_dim']['indices'][0:n_dims_]
                DataType_ = np.dtype([
                    ('head_tmp', 'S102400'),
                    ('data', endian + 'f8', n_indices_)
                ])
                self = np.fromfile(filename, dtype=DataType_)[0]['data'].view(cls)
                self.indices = {}
                for i in range(n_dims_):
                    self.indices[typename_[i]] = indices_[i][0:n_indices_[i]]
                self.type = typename_
                self.index = {}
                for i in range(n_dims_):
                    self.index[typename_[i]] = i
                self.head_data = head_data_

        if init_method == 'Var_file':
            if name == '':
                print('wrong ini params')
                exit(-1)
            else:
                head_data_ = np.load(name + '_meta.npy')
                n_dims_ = head_data_['head']['n_dims']
                type_ = head_data_['head']['one_dim']['type'][0:n_dims_]
                typename_ = [typename[i] for i in type_]
                n_indices_ = tuple(head_data_['head']['one_dim']['n_indices'][0:n_dims_])
                indices_ = head_data_['head']['one_dim']['indices'][0:n_dims_]

                self = np.load(name + '.npy').view(cls)
                self.indices = {}
                for i in range(n_dims_):
                    self.indices[typename_[i]] = indices_[i][0:n_indices_[i]]
                self.type = typename_
                self.index = {}
                for i in range(n_dims_):
                    self.index[typename_[i]] = i
                self.head_data = head_data_

        self.name               = name
        self.tree               = tree
        self.eventNumber        = eventNumber
        self.eventWeight        = eventWeight
        self.numberOfBins       = numberOfBins
        self.binningLogicSystem = binningLogicSystem

        return self

    def find_name(self, name_=''):
        if name_ in self.type:
            return self.index[name_]
        else:
            return -1

    def jack(self):
        index_ = self.find_name("conf")
        if index_ == -1:
            print("no index conf")
            exit(-1)
        return do_jack(self,index_)

    def eff_mass_log(self):
        index_ = self.find_name("t")
        if index_ == -1:
            print("no index conf")
            exit(-1)
        return do_jack(self,index_)

    def save(self):
        np.save(self.name, self)
        np.save(self.name+'_meta', self.head_data)

    def update_meta(self): #TODO
        print('to do')
