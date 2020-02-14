import streamlit as st
import pandas as pd
import time
import Session_State as ss
from datetime import datetime

SL = ss.get(Multichain=[],Problems=[],Algos=[],Members=[],P={},Status='')

st.header('Uncle Erdos : A Depiction')

st.sidebar.header('Panel')

ID = st.sidebar.text_input('Enter ID:',' ')
Problem = st.sidebar.text_input('Enter name of Problem',' ')
Algo = st.sidebar.text_input('Enter name of Algorithm :',' ')

def Chain(C_No,Problem_name,User_ID,Algorithm):
    if SL.Status == '10':
        if C_No == 1:
            Index = 0
        else:
            Index = len(SL.Problems)-1
        SL.Multichain.append({})
        now = datetime.now()
        Timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
        SL.Multichain[Index][SL.Problems[Index]] = {Algorithm : [[C_No,User_ID,Timestamp]]}       
    elif SL.Status == '01' or SL.Status == '00':
        Index = SL.Problems.index(Problem_name)
        if SL.Status == '01':
            now = datetime.now()
            Timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
            SL.Multichain[Index][SL.Problems[Index]][Algorithm] = [[C_No,User_ID,Timestamp]]
        else:
            now = datetime.now()
            Timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
            SL.Multichain[Index][SL.Problems[Index]][Algorithm].append([C_No,User_ID,Timestamp])
    
    #st.write(SL.Multichain[Index][SL.Problems[Index]])        
    #Prog_No = str(Index) + str(list(SL.Multichain[Index][Problem_name].keys()).index(Algorithm)) + str(len(SL.Multichain[Index][Algorithm])-1)        
    Prog_No = ''
    return Index,Prog_No        


def Interact():
    SL.Status = ''
    if ID in SL.Members:
        if Problem == ' ' or Algo == ' ':
            exit(0)
    else:
        SL.Members.append(ID)    
    if Problem not in SL.Problems:
        SL.Problems.append(Problem)
        if Algo not in SL.Algos:
            SL.Algos.append(Algo)
        SL.Status += '10'
    else:
        if Algo not in SL.Algos:
            SL.Algos.append(Algo)
            SL.Status += '01'
        else:
            SL.Status += '00'                
    X,Y = Chain(len(SL.Problems),Problem,ID,Algo)
    return X,Y


if st.sidebar.button('Okay'):
    i,PNo = Interact()
    st.sidebar.subheader(SL.Problems)
    st.sidebar.subheader(SL.Members)
    st.sidebar.subheader(SL.Algos)
    
    for i in range(len(SL.Problems)):
        st.subheader('Chain'+str(i+1))
        df = pd.DataFrame(SL.Multichain[i])
        st.table(df)
        st.sidebar.subheader('Chain'+str(i+1))
        SL.P[PNo] = st.sidebar.progress(0)
        for i in range(11):
            SL.P[PNo].progress(i**2)
            time.sleep(1)
    #    st.write(SL.Multichain[i])
    


