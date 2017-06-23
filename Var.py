import numpy as np
import os.path
import copy


def mass_eff_cosh(src, index_t):
    return np.arccosh((np.roll(src, +1, index_t)+np.roll(src, -1, index_t))/(2.*src))


def mass_eff_log(src, index_t):
    return np.log(np.roll(src, +1, index_t)/src)


def get_std_error(src, index_conf):
    return np.std(src, index_conf)/np.sqrt(src.shape[index_conf]-1.)


def do_jack(src, index_conf):
    tmp = np.sum(src, index_conf, keepdims=True)
    return (tmp-src)/(src.shape[index_conf]-1)


def get_jack_error(src, index_conf):
    return np.std(src, index_conf) * np.sqrt(src.shape[index_conf]-1)


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


def mod_head_name(head=None, index=0, new_name=''):
    n_dims_ = head['head']['n_dims']
    head['head']['one_dim']['type'][0:n_dims_][index] = np.where(typename == new_name)[0][0]


def mod_head_dim_size(head=None, index=0, new_size=0):
    n_dims_ = head['head']['n_dims']
    head['head']['one_dim']['n_indices'][0:n_dims_][index] = new_size


def mod_head_indices(head=None, index=0, indices=np.array([0])):
    n_dims_ = head['head']['n_dims']
    head['head']['one_dim']['n_indices'][0:n_dims_][index] = indices.size
    for i in range(indices.size):
        head['head']['one_dim']['indices'][0:n_dims_][index][i] = indices[i]


def rm_head_index(head_=None, index_=0):
    n_dims_ = head_['head']['n_dims']
    head_tmp = copy.deepcopy(head_)
    head_['head']['n_dims'] = n_dims_-1
    count = 0
    for i in range(0, n_dims_-1):
        if count == index_:
            count += 1
        head_['head']['one_dim'][i] = head_tmp['head']['one_dim'][count]
        count += 1


typename = np.array(["other", "x", "y", "z", "t", "d", "c", "d2",
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
                "t3", "t4", "t_source", "t_current", "t_sink",
                "bootstrap", "nothing"])

HeadType = np.dtype(
    [('head',
      [('n_dims', 'i4'), ('one_dim',
                          [('type', 'i4'), ('n_indices', 'i4'), ('indices', 'i4', 1024)
                           ], 16)
       ])
     ]
)


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
        head_data               = None,
        init_method             = 'shape'
        ):

        init_methods = []
        # CHECK METHOD

        self = np.zeros(0).view(cls)

        if init_method == 'shape': # init from shape
            self = np.zeros(shape=shape).view(cls)
            self.head_data = np.zeros(1,dtype=HeadType)[0]
            self.head_data['head']['n_dims'] = len(shape)
            for i in range(len(shape)):
                self.head_data['head']['one_dim']['n_indices'][i] = shape[i]
            self.update_meta()

        if init_method == 'array data': # init from list
            if data is not None and head_data is not None:
                # FIXME # TODO
                self = np.array(data).view(cls)
                self.head_data = head_data
                self.update_meta()
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

                tmp_ = Var(filename=file_names_[0], init_method="iog_file")
                shape_ = np.array(tmp_.shape)
                shape_[tmp_.find_name(name_=scatter_index_name)] = len(file_names_)
                self = np.zeros(shape=shape_).view(cls)

                self.head_data = tmp_.head_data
                n_dims_ = self.head_data['head']['n_dims']

                self.head_data['head']['one_dim']['n_indices'][0:n_dims_]\
                    [tmp_.find_name(name_=scatter_index_name)] = len(file_names_)

                for i in range(len(file_names_)):
                    tmp_ = Var(filename=file_names_[i], init_method="iog_file")
                    self[i, ...] = tmp_
                    self.head_data['head']['one_dim']['indices'][0:n_dims_ ]\
                        [tmp_.find_name(name_=scatter_index_name)][i] = tmp_.indices[scatter_index_name][0]

                self.update_meta()

        if init_method == 'scatter_Var_file':
            if scatter_file_name is None or scatter_num is None or scatter_index_name is None:
                print('wrong ini params')
                exit(-1)
            else:
                file_names_ = []
                for ic in range(scatter_num[0], scatter_num[1], scatter_num[2]):
                    if os.path.isfile(scatter_file_name % ic):
                        file_names_.append(scatter_file_name % ic)

                tmp_ = Var(name=file_names_[0], init_method="Var_file")
                shape_ = np.array(tmp_.shape)
                shape_[tmp_.find_name(name_=scatter_index_name)] = len(file_names_)
                self = np.zeros(shape=shape_).view(cls)

                self.head_data = tmp_.head_data
                n_dims_ = self.head_data['head']['n_dims']

                self.head_data['head']['one_dim']['n_indices'][0:n_dims_]\
                    [tmp_.find_name(name_=scatter_index_name)]= len(file_names_)

                for i in range(len(file_names_)):
                    tmp_ = Var(name=file_names_[i], init_method="Var_file")
                    self[i, ...] = tmp_
                    self.head_data['head']['one_dim']['indices'][0:n_dims_] \
                        [tmp_.find_name(name_=scatter_index_name)][i] = tmp_.indices[scatter_index_name][0]

                self.update_meta()

        if init_method == 'iog_file':
            if filename == '':
                print('wrong ini params')
                exit(-1)
            else:
                self.head_data = np.fromfile(filename, dtype=HeadType, count=1)[0]
                DataType_ = np.dtype([
                    ('head_tmp', 'S102400'),
                    ('data', endian + 'f8', tuple(self.head_data['head']['one_dim']['n_indices']\
                                                      [0:self.head_data['head']['n_dims']]))
                ])
                self = np.fromfile(filename, dtype=DataType_)[0]['data'].view(cls)
                self.head_data = np.fromfile(filename, dtype=HeadType, count=1)[0]
                self.update_meta()

        if init_method == 'Var_file':
            if name == '':
                print('wrong ini params')
                exit(-1)
            else:
                self = np.load(name + '.npy').view(cls)
                self.head_data = np.load(name + '_meta.npy')
                self.update_meta()

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
        new_head = self.head_data #FIXME #TODO
        mod_head_name(new_head,index_,'jackknife')
        return Var(data=do_jack(self,index_),head_data = new_head, init_method="array data")

    def eff_mass_log(self):
        index_ = self.find_name("t")
        if index_ == -1:
            print("no index conf")
            exit(-1)
        new_head = copy.deepcopy(self.head_data) #FIXME #TODO
        return Var(data=mass_eff_log(self,index_), head_data=new_head, init_method="array data")

    def eff_mass_cosh(self):
        index_ = self.find_name("t")
        if index_ == -1:
            print("no index conf")
            exit(-1)
        return mass_eff_cosh(self,index_)

    def get_std_error(self):
        index_ = self.find_name("conf")
        if index_ == -1:
            print("no index conf")
            exit(-1)
        return get_std_error(self,index_)

    def get_jack_error(self):
        index_ = self.find_name("jackknife")
        if index_ == -1:
            print("no index conf")
            exit(-1)
        new_head = copy.deepcopy(self.head_data) #FIXME #TODO
        rm_head_index(new_head,index_)
        return Var(data=get_jack_error(self,index_),head_data=new_head,init_method="array data")

    def get_jack_ave(self):
        index_ = self.find_name("jackknife")
        if index_ == -1:
            print("no index conf")
            exit(-1)
        new_head = copy.deepcopy(self.head_data) #FIXME #TODO
        rm_head_index(new_head,index_)
        return Var(data=np.sum(self,index_)/self.shape[index_],head_data=new_head,init_method="array data")

    def save(self):
        np.save(self.name, self)
        np.save(self.name+'_meta', self.head_data)

    def update_meta(self): #TODO
        n_dims_ = self.head_data['head']['n_dims']
        type_ = self.head_data['head']['one_dim']['type'][0:n_dims_]
        typename_ = [typename[i] for i in type_]
        n_indices_ = tuple(self.head_data['head']['one_dim']['n_indices'][0:n_dims_])
        indices_ = self.head_data['head']['one_dim']['indices'][0:n_dims_]
        self.indices = {}
        for i in range(n_dims_):
            self.indices[typename_[i]] = indices_[i][0:n_indices_[i]]
        self.type = typename_
        self.index = {}
        for i in range(n_dims_):
            self.index[typename_[i]] = i
