import numpy as np
MEM = [0.3, 0.5820596829288368, 0.6213070029394455, 0.6261765111982598, 0.6267311443153933, 0.6267932602722506, 0.6268144107235621, 0.6268237907185084, 0.6268059125543467, 0.6268381817304196, 0.626826298775474]
EM =[0.3, 0.5512828059742375, 0.6131119318182763, 0.6245128161116744, 0.6264382430975102, 0.6267581767201758, 0.6268110887608314, 0.6268198022928538, 0.6268213208527162, 0.6268214340803788, 0.6268214744081205] 
MEM = np.array(MEM)
EM = np.array(EM)
accu = 0.6268215
diff_EM_old = 1
diff_MEM_old =1

def get_string(number):
    return "&{0:.6f}".format(number)

for i in range(6):
    index = i
    theta_EM = EM[i]
    theta_MEM = MEM[i]
    diff_EM_New = np.abs(EM[i] - accu)
    diff_MEM_New = np.abs(MEM[i] - accu)
    diff_EM_ratio = diff_EM_New/diff_EM_old
    diff_MEM_ratio = diff_MEM_New/diff_MEM_old

    diff_EM_old = diff_EM_New
    diff_MEM_old = diff_MEM_New
    
    mylist = [index , theta_EM,  diff_EM_New, diff_EM_ratio, theta_MEM,  diff_MEM_New, diff_MEM_ratio]
    mystring = [get_string(number) for number in mylist]
    mystring[0] = str(index)
    mystring.append("\\\\ \hline")
    final_string =''
    for s in mystring:
        final_string = final_string+' '+s
    print final_string
    

