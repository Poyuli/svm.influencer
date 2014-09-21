def svm_read_csv(data_file_name):
    """
    svm_read_csv(data_file_name) -> [y, x, h]

    Read data from data_file_name.csv
    and return labels y, data instances x, and the header h with LIBSVM format.
    """
    
    prob_y = []
    prob_x = []
    
    f = open(data_file_name).readlines()
    
    # Store the header
    header = f.pop(0)
    for line in f:
        line = line.split(",", 1)
        
        # In case an instance with all zero features
        if len(line) == 1: line += ['']
        label, features = line
        xi = {}
        ind = 1
        for e in features.split(","):
            val = e
            xi[int(ind)] = float(val)
            ind += 1
        if float(label) == 0:
            label = -1
        prob_y += [float(label)]
        prob_x += [xi]
    return (prob_y, prob_x, header)
 
def write2txt(prob_x, prob_y, file_name):
    """
    Write LIBSVM-format data into the file.
    prob_x are data instances and y are labels.
    """
    
    txt = open(file_name, "w+")
    for i in range(0, len(prob_x)):
        raw_str = str(prob_y[i]) + " " + str(prob_x[i])[1:-1] + "\n"
        new_str = raw_str.replace(",","")
        new_str = new_str.replace(": ",":")
        txt.write(new_str)
    txt.close()



        


