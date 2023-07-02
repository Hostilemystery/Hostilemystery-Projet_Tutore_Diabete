import subprocess
import time
import streamlit as st

def main():
    delay_seconds = 0.6

    st.image("diabetesgif.gif", use_column_width=True)
    time.sleep(delay_seconds)

    subprocess.Popen(["streamlit", "run", "diabetes.py"], close_fds=True)
    
    
if __name__ == "__main__":
    main()