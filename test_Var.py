from Var import *

d = Var()
print(d.shape)
a = Var(shape=(2,2))
b = Var(shape=(2,2,2))
a = b
print(a.shape)

a = Var(filename="rbc_conf_2464_m0.005_0.04_000495_hyp.2pt.dov.glue.data")
print(a.shape)
print(a.indices['momentum'])
print(a.indices[a.type[a.index['momentum']]])
print(a.type)
print(a.index['momentum'])
print(a.find_name(name_='momentum'))
print(a[0,0,0,0,1,0,0])
print(a.flat[0])
print(a.flat[-1])

b = Var(scatter_file_name="rbc_conf_2464_m0.005_0.04_%06d_hyp.2pt.dov.glue.data", scatter_num=(495,535+1,40),
        scatter_index_name="conf",name='test')
print(b.shape)
print(b.indices['conf'])
print(b[0,0,0,0,1,0,0])
print(b.flat[-1])
print(b.head_data['head']['one_dim']['n_indices'][0:8])
b.save()
c = Var(name='test',load=True)
print(np.nonzero(b-c))
