
import streamlit as st

#import sys
#sys.path.insert(0, "../")
import primes.miller_rabin as pmr
import primes.baillie_psw as bpsw
import primes.quadratic_frobenius as qb
import primes.solovay_strassen as ss
import primes.strong_lucas as strluc

def prime_printer(num,boolval):
    if boolval:
        st.write(num,"might be prime!")
    else:
        st.write(num,"is composite!")


main_option = st.selectbox("Choose Functionality",["Prime Testing", "Integer Facotrisation"])

if main_option == 'Prime Testing':
    num_input = st.text_input("Please enter the number for primality check",5)
    algo = st.sidebar.selectbox("Choose Algorithm",["BPSW", "Miller Rabin","Quad_Forbenius","Solvay Strassen","Strong Lucas"])
    num = int(num_input)
    if st.button("Check!"):
        if algo == "Strong Lucas":
            prime_printer(num,strluc.is_prime(num))
        elif algo == "BPSW":
            prime_printer(num,bpsw.is_prime(num))
        elif algo == "Quad_Forbenius":
            prime_printer(num,qb.is_prime(num))
        elif algo == "Solvay Strassen":
            prime_printer(num,ss.is_prime(num))
        else:
            prime_printer(num,pmr.is_prime(num))

