import numpy as np

class Var(np.ndarray):

    def __new__(
        cls,
        name                    = "trk_pt",
        tree                    = None, # tree object
        eventNumber             = None,
        eventWeight             = None,
        numberOfBins            = None, # binning
        binningLogicSystem      = None, # binning
        shape                   = None,
        filename                = '',
        conf_names              = None,
        conf_num                = None,
        endian                  = '<',
        data                    = None
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

        self = np.zeros(shape=shape).view(cls)

        if data is not None:
            self = np.array(data).view(cls)

        if conf_names is not None and conf_num is not None:
            file_names_ = []
            for ic in range(conf_num[0], conf_num[1], conf_num[2]):
                file_names_.append(conf_names%ic)

            tmp_ = Var(filename=file_names_[0])
            shape_ = np.array(tmp_.shape)
            shape_[tmp_.find_name(name_='conf')] = len(file_names_)
            self = np.zeros(shape=shape_).view(cls).view(cls)

            head_data_ = np.fromfile(file_names_[0], dtype=HeadType, count=1)[0]
            n_dims_ = head_data_['head']['n_dims']
            type_ = head_data_['head']['one_dim']['type'][0:n_dims_]
            typename_ = [typename[i] for i in type_]
            n_indices_ = head_data_['head']['one_dim']['n_indices'][0:n_dims_]
            indices_ = head_data_['head']['one_dim']['indices'][0:n_dims_]

            n_indices_[tmp_.find_name(name_='conf')] = len(file_names_)

            for i in range(len(file_names_)):
                tmp_ = Var(filename=file_names_[i])
                self[i,...] = tmp_
                indices_[tmp_.find_name(name_='conf')][i] = tmp_.indices['conf'][0]

            self.indices = {}
            for i in range(n_dims_):
                self.indices[typename_[i]] = indices_[i][0:n_indices_[i]]
            self.type = typename_
            self.index = {}
            for i in range(n_dims_):
                self.index[typename_[i]] = i

        if filename != '':
            head_data_ = np.fromfile(filename, dtype=HeadType, count=1)[0]
            n_dims_ = head_data_['head']['n_dims']
            type_ = head_data_['head']['one_dim']['type'][0:n_dims_]
            typename_ = [typename[i] for i in type_]
            n_indices_ = tuple(head_data_['head']['one_dim']['n_indices'][0:n_dims_])
            indices_ = head_data_['head']['one_dim']['indices'][0:n_dims_]
            DataType_ = np.dtype([
            ('head_tmp','S102400'),
            ('data',endian+'f8',n_indices_)
            ])
            self = np.fromfile(filename, dtype=DataType_)[0]['data'].view(cls)
            self.indices = {}
            for i in range(n_dims_):
                self.indices[typename_[i]] = indices_[i][0:n_indices_[i]]
            self.type = typename_
            self.index = {}
            for i in range(n_dims_):
                self.index[typename_[i]] = i

        self._name              = name
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
        #return jack(self,index_)
