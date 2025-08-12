import utils.mrass as mrass

if __name__ == "__main__":
    thread_list  = [123,150,300]
    c = 20
    fi_funcs = [0.3,0.4,0.5]
    allocation,time = mrass.mrass_allocate(thread_list,c,fi_funcs)
    print(f"allocation = {allocation}\ntime = {time}")