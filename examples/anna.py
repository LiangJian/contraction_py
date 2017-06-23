from Var import *
import os.path
import time
import warnings
warnings.filterwarnings("ignore")

reload = False

start = time.time()

if os.path.isfile('./anna/2pfs_mv1_ts1.npy') and not reload:
    twopf = Var(name='./anna/2pfs_mv1_ts1',init_method='Var_file')
else:
    twopf = Var(scatter_file_name="./corr_low/rbc_conf_4896_m0.00078_0.0362_%06d_hyp_cpv_2pt.Nmv01.ts01", scatter_num=(425,2185+1,20),
                scatter_index_name="conf",name='./anna/2pfs_mv1_ts1',init_method='scatter_iog_file')
    twopf.save()

twopf_jack = twopf.jack()
twopf_jack_head = twopf_jack.head_data
twopf_jack = twopf_jack[:,:, 1:16:+1,:]+\
             twopf_jack[:,:,31:16:-1,:]+\
             twopf_jack[:,:,33:48:+1,:]+\
             twopf_jack[:,:,63:48:-1,:]+\
             twopf_jack[:,:,65:80:+1,:]+\
             twopf_jack[:,:,95:80:-1,:]
twopf_jack /= 6.
mod_head_dim_size(twopf_jack_head,twopf.find_name('t'),15)
twopf_jack = Var(data=twopf_jack,head_data=twopf_jack_head,init_method="array data")
twopf_jack_ave = twopf_jack.get_jack_ave()
twopf_jack_err = twopf_jack.get_jack_error()
mass_jack = twopf_jack.eff_mass_log()
mass_jack_ave = mass_jack.get_jack_ave()
mass_jack_err = mass_jack.get_jack_error()

for it in range(0,15):
    print('%03d'%it,'\t%8.6e'%twopf_jack_ave[0,it,0],'\t%8.6e'%twopf_jack_err[0,it,0],'\t%8.6e'%mass_jack_ave[0,it,0],'\t%8.6e'%mass_jack_err[0,it,0])

if os.path.isfile('./anna/3pfs_mv1_ml2_ts1.npy') and not reload:
    threepf = Var(name='./anna/3pfs_mv1_ml2_ts1',init_method='Var_file')
else:
    threepf = Var(scatter_file_name="./corr_low/rbc_conf_4896_m0.00078_0.0362_%06d_hyp_cpv_3pt.Nmv01.Nml02.ts01", scatter_num=(425,2185+1,20),
                scatter_index_name="conf",name='./anna/3pfs_mv1_ml2_ts1',init_method='scatter_iog_file')
    threepf.save()

threepf_jack = threepf.jack()
threepf_jack_head = threepf_jack.head_data
threepf_jack = threepf_jack[:,:,:,1:16,1:16,:,:,:]\
             + threepf_jack[:,:,:,31:16:-1,31:16:-1,:,:,:]\
             + threepf_jack[:,:,:,33:48,33:48,:,:,:]\
             + threepf_jack[:,:,:,63:48:-1,63:48:-1,:,:,:]\
             + threepf_jack[:,:,:,65:80,65:80,:,:,:]\
             + threepf_jack[:,:,:,95:80:-1,95:80:-1,:,:,:]
threepf_jack /= 6.
for i in range(1, threepf.shape[threepf.find_name('displacement')]):
    threepf_jack[:,:,:,:,:,:,i,:] += threepf_jack[:,:,:,:,:,:,i-1,:]

mod_head_dim_size(threepf_jack_head,threepf.find_name('t'),15)
mod_head_dim_size(threepf_jack_head,threepf.find_name('t2'),15)
threepf_jack = Var(data=threepf_jack,head_data=threepf_jack_head,init_method="array data")
threepf_jack_ave = threepf_jack.get_jack_ave()
threepf_jack_err = threepf_jack.get_jack_error()

ratio_jack = Var(data=threepf_jack/twopf_jack.reshape((54, 1, 1, 15, 1, 1, 1, 2)),head_data=threepf_jack_head,init_method="array data")
ratio_jack_ave = ratio_jack.get_jack_ave()
ratio_jack_err = ratio_jack.get_jack_error()

ratio_jack_sum = ratio_jack

end = time.time()
print("%3.2f"%(end-start)," s used.")
