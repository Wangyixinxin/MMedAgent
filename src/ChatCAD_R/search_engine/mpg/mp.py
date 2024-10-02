from log import info,warn,save_log
import os,sys,time
import multiprocessing

class MP:
    """ MP class is for fast data processing with multi cpus used together, based on python multiprocessing package. Useful & easy.
    Usage:
        1. Initlize an multiprocessing pool. e.g.: `my_mp=MP()`
        2. Append your own function and parameters to the MP pool. e.g.: `my_mp.append(numpy.add,1,2)`. The first parameters is your function that will execute with a new pthread. The later parameters are the parameters to be given to your function. The only concern is that your function should be non-local, and the function object name can't be changed during your program. It would be best a global function. 
        3. Repeat step 2, till all functions that you want to execute has been added to the multiprocessing pool. It's probably done with a FOR statement.
        4. You can do anything now. All the functions will be running in back-end. When a function finished, it will show logs to the screen, either execute successfully, or exceptions happened.
        5. Before your program end, get the result of function. e.g.: `return_table=my_mp.ret()` It will return a list for all have been execute, including function name, parameters, and the execute result or exceptions. If some function hasn't been finished, the program will keep waiting here.
        ## TODO: A concern is that the random state in parallel execute function is very strange. It may affect some behaviour of numpy. You'd better set random variables outside, or set random state manually in your function.
        # TODO: Check multi node can be easily used on the hpc. If not, add flag control to fit run with N nodes together.
        # TODO: add CSV file log output during running MPG.
        # TODO: add stdout and stderr logging record.
    """
    @staticmethod
    def time_wapper(func,*args,**kwargs):
        start_time=time.time()
        try:
            res=func(*args,**kwargs)
        except Exception as e:
            e.exec_time=time.time()-start_time
            raise e
        return (time.time()-start_time,res)
    def __init__(self,max_processor=56,save_log=True):
        self.max_processor=max_processor
        self.mulpool=multiprocessing.Pool(max_processor)
        self.exec_list=[]
        self.close=False
        self.save_log=save_log

    def append(self,func,*args,**kwargs):
        if self.close:
            raise Exception("MulPool has closed.")
        
        mulpool,exec_list=self.mulpool,self.exec_list
        tid=len(exec_list)
        exec_list.append({"func":func.__name__,"args":args,"kwargs":kwargs})
        # print(len(args))

        def succ_callback(dt_res):
            dtime,res=dt_res
            self.exec_list[tid]["exec_time"]=dtime
            self.exec_list[tid]["res"]=res
            info(f"{func.__name__} finished in {dtime}.")

        def err_callback(e):
            dtime=e.exec_time
            self.exec_list[tid]["exec_time"]=dtime 
            self.exec_list[tid]["err"]=e
            warn(f"Exception happen in {func.__name__} in {dtime}!!")
            warn(repr(e))

        mulpool.apply_async(func=MP.time_wapper,args=(func,*args),kwds=kwargs,callback=succ_callback,error_callback=err_callback)

    def wait(self):
        if not self.close:
            self.close=True
            mulpool=self.mulpool
            exec_list=self.exec_list
            mulpool.close()
            mulpool.join()

    def ret(self):
        self.wait()
        ret=self.exec_list
        if self.save_log:
            save_log(ret)
        return ret