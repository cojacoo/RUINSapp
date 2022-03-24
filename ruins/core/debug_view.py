import streamlit as st
from ruins.core import Config, DataManager

def debug_view(config: Config, dataManager: DataManager):
    '''
    Set Config['debug'] = 'True' to display debug view which
    shows current Config and dataManager parameters.
    '''
    st.title('Debug view')
    
    left, right = st.columns(2)

    left.write('Config paramters')
    left.markdown(config.__dict__) # TODO: what to display exactly?

    right.write('DataManager parameters')
    right.markdown(dataManager) # TODO: what to display exactly?

    st.stop()