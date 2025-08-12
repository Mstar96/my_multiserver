import utils.mrass as mrass

if __name__ == "__main__":
    thread_list  = [123,150,300]
    c = 20
    fi_funcs = [0.3,0.5,0.4]
    mrass.mrass_allocate(thread_list,c,fi_funcs)