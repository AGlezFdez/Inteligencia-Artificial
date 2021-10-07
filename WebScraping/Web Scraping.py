#!/usr/bin/env python
# coding: utf-8

# In[22]:


from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

driver = webdriver.Chrome(ChromeDriverManager().install())

url="https://www1.sedecatastro.gob.es/CYCBienInmueble/OVCBusqueda.aspx"

driver.get(url)

coord = driver.find_element_by_id("tabcoords")
coord.click()


# In[23]:


cart = driver.find_element_by_id("taburbana")
cart.click()


# In[24]:


rc = driver.find_element_by_name("ctl00$Contenido$txtRC2")

codigo = "9872023 VH5797S 0001 WX"
rc.send_keys(codigo)


# In[25]:


cart = driver.find_element_by_name("ctl00$Contenido$btnNuevaCartografia")
cart.click()


# In[ ]:




