#!/usr/bin/env python
# coding: utf-8

# In[57]:


from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import Select

driver = webdriver.Chrome(ChromeDriverManager().install())

url="https://www.escapadarural.com/"

driver.get(url)


# In[58]:


loc = driver.find_element_by_class_name('ui-autocomplete-input')
loc.send_keys('Benasque')


# In[59]:


hab = Select(driver.find_element_by_id('fullOrRooms'))
hab.select_by_value('full')


# In[60]:


per = Select(driver.find_element_by_id('peopleSearch'))
per.select_by_value('2')


# In[61]:


but = driver.find_element_by_id('buttonSearch').click()


# In[52]:


get_url = driver.current_url
print(get_url)


# In[53]:


from bs4 import BeautifulSoup as bs
import requests

url = get_url
response = requests.get(url)
html = response.content


# In[54]:


soup = bs(html, "lxml")


# In[55]:


soup.title


# In[56]:


file = open('read.txt', 'w')
all_h3 = soup.find_all("h3", class_="c-h3--result")
for h3 in all_h3:
       print(h3.get_text(strip=True))
       file.write(h3.get_text(strip=True) + '\n')
file.close()


# In[ ]:




