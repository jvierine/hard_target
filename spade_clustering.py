import glob
import h5py
import numpy as n
import matplotlib.pyplot as plt
import stuffr
import scipy.interpolate as sint

def get_mf_results(dirname,
                   amb_len_lim=[120,160], 
                   plot_good=False,
                   plot_debug=False):
    
    fl=glob.glob("%s/*/gmf*.h5"%(dirname))
    fl.sort()
    time=n.array([],dtype=n.float64)
    doppler_ms=n.array([],dtype=n.float64)
    range_km=n.array([],dtype=n.float64)
    snr=n.array([],dtype=n.float64)
    
    for f in fl:
        print(f)
        h=h5py.File(f,"r")
        gdc=h["gmf_dc"][()]
        print(gdc.shape)
        print(gdc)
        g=h["gmf"][()]
        print(g.shape)
        print(g)
        plt.pcolormesh(10.0*n.log10(g),vmin=0,vmax=20)
        plt.show()
        h.close()

    length=length*1e6
    if plot_debug:
        plt.plot(range_km,psl,".")
        plt.show()
        plt.plot(doppler_ms,psl,".")
        plt.show()
        
        plt.plot(time,psl,".")
        plt.show()

    gidx=n.where( (length > amb_len_lim[0]) &
                  (length < amb_len_lim[1]) &
                  (psl > psl_lim) 

                  )[0]

    # low doppler and low psl is most likely space debris
    bidx=n.where( (n.abs(doppler_ms) < psl_dop) & (psl < psl_dop_lim) )[0]
    gidx=n.setdiff1d(gidx,bidx)
    
#    time=time[gidx]
 #   doppler_ms=doppler_ms[gidx]
  #  range_km=range_km[gidx]
   # snr=snr[gidx]
    #psl=psl[gidx]
   # length=length[gidx]
   # phase=phase[gidx]
    #cross_spectra=cross_spectra[gidx]

    if plot_good:
        plt.plot(time,range_km,".")
        plt.plot(time[gidx],range_km[gidx],".")    
        plt.show()
        plt.plot(time,doppler_ms,".")
        plt.plot(time[gidx],doppler_ms[gidx],".")    
        plt.show()
        plt.plot(time,10.0*n.log10(snr),".")
        plt.plot(time[gidx],10.0*n.log10(snr[gidx]),".")    
        plt.show()

    # create error functions
    db=[0,200]
    r_std_fun=sint.interp1d(db,[0.03,0.03])
    v_std_fun=sint.interp1d(db,[0.3,0.3])    

    
    return({"time":time[gidx],"range":range_km[gidx],"snr":snr[gidx],"doppler":doppler_ms[gidx],"phase":phase[gidx],"xspec":cross_spectra[gidx],"r_std_fun":r_std_fun,"v_std_fun":v_std_fun})


def fit_traj(r_meas,
             dop_meas,
             time,
             snr,
             range_std0=0.02,
             max_vel_resid=2,
             max_range_resid=0.150,
             debug_resid=False,
             debug_fit=False,
             output_dir="",
             vel_std0=0.170):
    
    t0=time[n.argmax(snr)]
    N=len(r_meas)
    A=n.zeros([2*len(r_meas),3])
    m=n.zeros(2*len(r_meas))

    delta_t=n.abs(time-t0)

    # std increases when going away from central point
    range_std=range_std0 + delta_t
    # std increases when going away from central point
    vel_std=vel_std0 + delta_t
    
    A[0:N,0]=1.0/range_std 
    A[0:N,1]=(time-t0)/range_std
    A[0:N,2]=(0.5*(time-t0)**2.0)/range_std
    
    A[N:(2*N),1]=1.0*(1/vel_std)
    A[N:(2*N),2]=(time-t0)*(1/vel_std)

    m[0:N]=r_meas/range_std
    m[N:(2*N)]=dop_meas/vel_std

    # step 1
    xhat=n.linalg.lstsq(A,m)[0]
    print(xhat)
    model=n.dot(A,xhat)
    # force positive deceleration
    if xhat[2] < 0:
        xhat[2]=0.0
    resid = model-m
    bidx=n.where( (n.abs(resid[0:N]*range_std) > max_range_resid) |  (n.abs(resid[N:(2*N)]*vel_std) > max_vel_resid) )[0]

    # step 2
    A2=n.copy(A)
    m2=n.copy(m)    
    A2[bidx,:]=0.0
    m2[bidx]=0.0
    A2[bidx+N,:]=0.0
    m2[bidx+N]=0.0

    xhat=n.linalg.lstsq(A2,m2)[0]
    # force positive deceleration
    if xhat[2] < 0:
        xhat[2]=0.0
    print(xhat)
    model=n.dot(A,xhat)
    resid = model-m
    bidx=n.where( (n.abs(resid[0:N]*range_std) > max_range_resid) |  (n.abs(resid[N:(2*N)]*vel_std) > max_vel_resid) )[0]

    # step 3
    A2=n.copy(A)
    m2=n.copy(m)    
    A2[bidx,:]=0.0
    m2[bidx]=0.0
    A2[bidx+N,:]=0.0
    m2[bidx+N]=0.0

    xhat=n.linalg.lstsq(A2,m2)[0]
    # force positive deceleration
    if xhat[2] < 0:
        xhat[2]=0.0
    print(xhat)
    model=n.dot(A,xhat)
    resid = model-m
    bidx=n.where( (n.abs(resid[0:N]*range_std) > max_range_resid) |  (n.abs(resid[N:(2*N)]*vel_std) > max_vel_resid) )[0]
    
    
    print(len(bidx))
    if debug_resid:
        plt.plot(resid[0:N]*range_std,".")
        
        plt.plot(resid[bidx]*range_std,".",color="red")    
        plt.show()
        plt.plot(resid[N:(2*N)]*vel_std,".") 
        plt.plot(resid[bidx+N]*vel_std,".",color="red")       
        plt.show()

    if debug_fit:
        plt.subplot(221)
        plt.plot(time-t0,r_meas,"x")
        plt.plot( (time-t0)[bidx],r_meas[bidx],"x",color="red")    
        plt.plot(time-t0,model[0:N]*range_std)
        plt.ylabel("Range (km)")
        plt.xlabel("Time (s)")
        
        plt.subplot(222)
        plt.plot( (time-t0),dop_meas,"x")    
        plt.plot( (time-t0)[bidx],dop_meas[bidx],"x",color="red")
        plt.plot(time-t0,model[N:(2*N)]*vel_std)
        plt.ylabel("Doppler (km/s)")
        plt.xlabel("Time (s)")
        
        plt.subplot(223)
        plt.plot(time-t0,(r_meas-model[0:N]*range_std)*1e3,"x")
        plt.ylabel("Range residual (m)")
        plt.xlabel("Time (s)")
        
        plt.subplot(224)
        plt.plot(time-t0,(dop_meas-model[N:(2*N)]*vel_std)*1e3,"x")
        plt.ylabel("Doppler residual (m/s)")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig("%s/fit-%1.2f.png"%(output_dir,t0))
        plt.clf()
        plt.close()

    good_idx=n.setdiff1d(n.arange(len(r_meas),dtype=int),bidx)

    range_resid=resid[0:N]*range_std*1e3
    doppler_resid=resid[N:(2*N)]*vel_std*1e3
    
    # return the indices with outliers removed
    return({"t0":t0,"xhat":xhat,"good_idx":good_idx,"range_residual":range_resid,"doppler_residual":doppler_resid})



def cluster(r,
            time_window=60.0,
            max_duration=3.0,
            max_range_offset=1.0,
            debug_selection=False,
            output_dir="",
            debug_all=False):

    """
    mark all meteors belonging to the same event with the same number
    """

    if debug_all:
        plt.subplot(211)
        plt.plot(r["time"],r["range"],".")
        plt.subplot(212)
        plt.plot(r["time"],r["doppler"],".")
        plt.show()

    snr_idx=n.argsort(r["snr"])[::-1]
    print(snr_idx)

    marked=n.zeros(len(r["time"]),dtype=int)
    # no meteor is -1, 0..N-1 indicates succesfully clustered meteor index
    marked[:]=-1
    print(marked)
    fit_ranges=[]
    fit_dopps=[]
    fit_times=[]        
    fit_tr={}
    meteor_num=0
    r["range_residual"]=n.zeros(len(marked))
    r["doppler_residual"]=n.zeros(len(marked))
    
    # give each meteor one chance to be associated with others
    # go in the order ot decreasing snr
    for idx in snr_idx:
        # if not associated yet with any event
        if marked[idx] == -1:
            t0=r["time"][idx]
            r0=r["range"][idx]
            v0=r["doppler"][idx]/1e3
            dt = r["time"]-t0
            delta_r = r["range"] - (r0 + (0.5*(v0+r["doppler"]/1e3))*dt)
            
            meteor_idx=n.where( (n.abs(delta_r) < max_range_offset) & (n.abs(dt) < 0.5*max_duration) & (marked == -1) )[0]

            traj_res=fit_traj(r["range"][meteor_idx],r["doppler"][meteor_idx]/1e3,r["time"][meteor_idx],r["snr"][meteor_idx],output_dir=output_dir)

            # todo try fitting at this stage, throw away outliers.
            if len(traj_res["good_idx"]) > 3:
                marked[meteor_idx[traj_res["good_idx"]]]=meteor_num
                fit_ranges.append(traj_res["xhat"][0])
                fit_dopps.append(traj_res["xhat"][1])
                fit_times.append(traj_res["t0"])
                fit_tr[meteor_num]=traj_res

                # copy residuals for further processing
                r["range_residual"][meteor_idx[traj_res["good_idx"]]]=traj_res["range_residual"][traj_res["good_idx"]]
                r["doppler_residual"][meteor_idx[traj_res["good_idx"]]]=traj_res["doppler_residual"][traj_res["good_idx"]]                
                
                meteor_num+=1

            if debug_selection:
                plt.subplot(311)            
                plt.scatter(r["time"]-t0,delta_r,c=marked,s=1)
                plt.plot(r["time"][meteor_idx]-t0,delta_r[meteor_idx],".",color="red")
                plt.xlim(-0.5,0.5)
                plt.ylim(-4,4)
                
                plt.subplot(312)
                plt.scatter(r["time"],r["range"],c=marked,s=1)
                plt.plot(r["time"][meteor_idx],r["range"][meteor_idx],".",color="red")
                plt.xlim(t0-0.5,t0+0.5)
                plt.subplot(313)
                plt.scatter(r["time"],r["doppler"],c=marked,s=1)
                plt.plot(r["time"][meteor_idx],r["doppler"][meteor_idx],".",color="red")
                plt.xlim(t0-0.5,t0+0.5)            
                plt.show()
    
    if debug_all:
        plt.subplot(211)
        plt.scatter(r["time"],r["range"],c=marked,s=1,cmap="prism")
        plt.xlabel("Time (unix seconds)")
        plt.ylabel("Range (km)")
        plt.subplot(212)
        plt.scatter(r["time"],r["doppler"]/1e3,c=marked,s=1,cmap="prism")
        plt.xlabel("Time (unix seconds)")
        plt.ylabel("Doppler (km/s)")
        plt.show()
    r["meteor_numbers"]=marked
    r["fits"]=fit_tr
    if debug_all:
        
        plt.subplot(121)
        plt.plot(fit_times,fit_ranges,".")
        
        plt.subplot(122)
        plt.plot(fit_times,fit_dopps,".")
        plt.show()
        
        plt.plot(fit_dopps,fit_ranges,".")
        plt.show()
    return(r)
                
def plot_meteor_events(r,output_dir,vel_std=0.15,r_std=0.01):
    
    metnums=n.unique(r["meteor_numbers"])
    for mn in metnums:
        if mn >= 0:
            idx=n.where(r["meteor_numbers"]==mn)[0]

            if len(idx) > 3:
                t0 = n.min(r["time"][idx])
                t1 = n.max(r["time"][idx])

                other_idx=n.where( (r["time"]>(t0-0.1)) & (r["time"]<(t1+0.1)) & (r["meteor_numbers"]!=mn) )[0]

                tv=n.linspace(t0,t1,num=1000)
                tmean=0.5*(t1+t0)
                plt.figure(figsize=(10,10))
                plt.subplot(321)
                plt.plot(r["time"][other_idx]-tmean,r["range"][other_idx],".",color="red")
#                plt.plot(r["time"][idx]-tmean,r["range"][idx],".")

                plt.errorbar(x=r["time"][idx]-tmean,y=r["range"][idx],yerr=r["r_std_fun"](10.0*n.log10(r["snr"][idx]))*2/1e3,linestyle='',marker=".")
                
#                plt.xlim(t0-tmean-0.1,t1-tmean+0.1)
                plt.xlabel("Time (s)")
                plt.ylabel("Range (km)")                
                delta_t=tv-r["fits"][mn]["t0"]
                plt.plot(tv-tmean,r["fits"][mn]["xhat"][0]+r["fits"][mn]["xhat"][1]*delta_t+0.5*r["fits"][mn]["xhat"][2]*delta_t**2.0,zorder=32)

                r_model=r["fits"][mn]["xhat"][0]+r["fits"][mn]["xhat"][1]*delta_t+0.5*r["fits"][mn]["xhat"][2]*delta_t**2.0
                r_model
                
                plt.title("Meteor %d"%(mn))
                
                plt.subplot(322)        
                plt.plot(r["time"][other_idx]-tmean,r["doppler"][other_idx]/1e3,".",color="red")
                
                plt.errorbar(x=r["time"][idx]-tmean,y=r["doppler"][idx]/1e3,yerr=r["v_std_fun"](10.0*n.log10(r["snr"][idx]))*2/1e3,linestyle='',marker=".")

                
                plt.plot(tv-tmean,r["fits"][mn]["xhat"][1]+r["fits"][mn]["xhat"][2]*delta_t,zorder=32)
                plt.xlabel("Time (s)")
                plt.ylabel("Doppler (km/s)")
                plt.title("%s"%(stuffr.unix2datestr(tmean)))
#                plt.xlim(t0-tmean-0.1,t1-tmean+0.1)
                
                plt.subplot(323)
                plt.plot(r["time"][other_idx]-tmean,10.0*n.log10(r["snr"][other_idx]),".",color="red")                        
                plt.plot(r["time"][idx]-tmean,10.0*n.log10(r["snr"][idx]),".")
                plt.xlabel("Time (s)")
                plt.ylabel("SNR (dB)")                
#                plt.xlim(t0-tmean-0.1,t1-tmean+0.1)                
                
                plt.subplot(324)
                plt.plot(r["time"][other_idx]-tmean,n.unwrap(n.angle(r["xspec"][other_idx])),".",color="red")  
                plt.plot(r["time"][idx]-tmean,n.unwrap(n.angle(r["xspec"][idx])),".")
                plt.xlabel("Time (s)")
                plt.ylabel("Cross-antenna phase (rad)")
#                plt.xlim(t0-tmean-0.1,t1-tmean+0.1)

                plt.subplot(325)
                plt.plot(r["time"][idx]-tmean,r["range_residual"][idx],".")
                plt.xlabel("Time (s)")
                plt.ylabel("Range residual (m)")                
#                plt.xlim(t0-tmean-0.1,t1-tmean+0.1)
                
                plt.subplot(326)
                plt.plot(r["time"][idx]-tmean,r["doppler_residual"][idx],".")
                plt.xlabel("Time (s)")
                plt.ylabel("Doppler residual (m/s)")
#                plt.xlim(t0-tmean-0.1,t1-tmean+0.1)                

                
                plt.tight_layout()
                plt.savefig("%s/fit-%1.2f.png"%(output_dir,t0))
                plt.clf()
                plt.close()

def estimate_errors(r,dB_step=0.5,dB_width=3.0,
                    debug_plot=True):


    
    gidx=n.where(r["meteor_numbers"]!=0)[0]

    snr_dB=10.0*n.log10(r["snr"][gidx])
    rr=r["range_residual"][gidx]
    dr=r["doppler_residual"][gidx]    

    n_steps=int(n.ceil((n.max(snr_dB)-n.min(snr_dB))/dB_step))
    min_dB=n.min(snr_dB)
    max_dB=n.max(snr_dB)
    rstds=[]
    drstds=[]
    db=[]        
    for i in range(n_steps):
        this_idx=n.where( (snr_dB > (i*dB_step + min_dB-dB_width)) & (snr_dB < ((i+1)*dB_step + min_dB+dB_width)))[0]
        rstds.append(n.std(rr[this_idx]))
        drstds.append(n.std(dr[this_idx]))
        db.append(i*dB_step+min_dB)
        #plt.hist(rr[this_idx],bins=20)
        #plt.show()
    if debug_plot:
        plt.subplot(221)
        plt.plot(db,rstds)
        plt.xlabel("SNR (dB)")
        plt.ylabel("Range stdev (m)")    
        plt.subplot(222)
        plt.plot(db,drstds)
        plt.ylabel("Doppler stdev (m/s)")
        plt.xlabel("SNR (dB)")
        
        plt.subplot(223)
        plt.hist2d(10.0*n.log10(r["snr"][gidx]),r["range_residual"][gidx],bins=[20,100],range=[[10,70],[-100,100]])

#        plt.plot(10.0*n.log10(r["snr"][gidx]),r["range_residual"][gidx],".",alpha=0.2)
        plt.ylabel("Range residual (m)")
        plt.xlabel("SNR (dB)")    
        plt.subplot(224)
#        plt.plot(10.0*n.log10(r["snr"][gidx]),r["doppler_residual"][gidx],".",alpha=0.2)
        plt.hist2d(10.0*n.log10(r["snr"][gidx]),r["doppler_residual"][gidx],bins=[20,100],range=[[10,70],[-1000,1000]])
        plt.ylabel("Doppler residual (m/s)")    
        plt.xlabel("SNR (dB)")
        plt.tight_layout()
        plt.savefig("errors.png")
        plt.close()
        plt.clf()

    db[0]=0
    db[len(db)-1]=db[len(db)-1]+100.0
    r["r_std_fun"] = sint.interp1d(db,rstds)
    r["v_std_fun"] = sint.interp1d(db,drstds)
    return(r)

def save_results(r,fname="test.h5"):
    fit_pars=[]
    fit_t0=[]    

    metnums=n.unique(r["meteor_numbers"])
    print(len(metnums))
    for mn in metnums:
        if mn >= 0:
            fit_pars.append(r["fits"][mn]["xhat"])
            fit_t0.append(r["fits"][mn]["t0"])            

    ho=h5py.File(fname,"w")
    ho["snr"]=r["snr"]
    ho["range_residual"]=r["range_residual"]
    ho["doppler_residual"]=r["doppler_residual"]
    ho["time"]=r["time"]
    ho["range"]=r["range"]
    ho["doppler"]=r["doppler"]/1e3
    ho["spec"]=r["xspec"]
    ho["meteor_numbers"]=r["meteor_numbers"]
    ho["fit_pars"]=fit_pars
    ho["fit_t0"]=fit_t0
    ho.close()

if __name__ == "__main__":
    r=get_mf_results("/data1/eiscat_leo/20210412/leo_bpark_2.1u_NO@uhf/drf/gmf5")
#    r=cluster(r,output_dir="/mnt/data/juha/test_output")
 #   print("error estimation")    
  #  r=estimate_errors(r)
 #   print("saving")
  #  save_results(r)
 #   print("plotting")
  #  plot_meteor_events(r,output_dir="/mnt/data/juha/test_output")
