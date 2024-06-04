from selenium import webdriver
from selenium.webdriver.common.by import By

from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService

import random
from time import sleep

# Define modifiable elements
modifiable_elements = ["input", "button", "select", "textarea"]

# Define display elements
display_elements = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "span", "a", "img", "div"]
display_classes = ["text-element"]

# URL of the webpage to scan
url = "http://192.168.100.202"

def find_modifiable(driver):
    # Find modifiable elements
    tmp = []
    modifiable = []
    for tag in modifiable_elements:
        elements = driver.find_elements(By.TAG_NAME, tag)
        tmp.extend(elements)

    for element in tmp:
        if element.is_displayed():
            modifiable.append(element)

    return modifiable

fill_values = {
    'Username': 'admin',
    'Password': 'aruba123'
}

def wait_js_queries(driver, repeat=60, delay=0.5):
    for _ in range(repeat):
        is_active_ajax_request = driver.execute_script(
            "return ($.active);")
        if is_active_ajax_request <= 0:
            break
        sleep(delay)
    else:
        raise RuntimeError(
            f'Error waiting JS queries to finish after {delay*repeat} seconds.')

def execute_action(action):
    if action.tag_name == 'button':
        action.click()
    elif action.tag_name == 'input':
        name = action.accessible_name
        value = fill_values[name]
        action.clear()
        action.send_keys(value)
    else:
        raise KeyError('Action is not supported')

def check_goal(driver):
    wait_js_queries(driver)
    try:
        driver.find_element(
            By.ID, "inlineBlockZoomDescription")
        return True
    except Exception:
        return False

# Initialize the Chrome webdriver
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

# Open the webpage
driver.get(url)

# Get the available actions
actions = find_modifiable(driver=driver)

# Show actions
print("Actions:")
for element in actions:
    is_editable = driver.execute_script("return arguments[0].readOnly === false && !arguments[0].disabled;", element)
    print(f'id: {element.id}')
    print(f'accessible_name: {element.accessible_name}')
    print(f'text: {element.text}')
    print(f'tag_name: {element.tag_name}')
    print(f'type: {element.get_attribute("type")}')
    print(f'is_displayed: {element.is_displayed()}')
    print(f'is_enabled: {element.is_enabled()}')
    print(f'value: {element.get_attribute("value")}')
    print()

# Try to reach the goal using a random approach
max_attempts = 20
for _ in range(max_attempts):

    # Perform a random action
    a = random.choice(actions)
    execute_action(a)
    print(f'Action executed: {a.accessible_name}')

    sleep(3)

    # Check target
    goal_reached = check_goal(driver)
    print(f'Goal reached: {goal_reached}')

    if goal_reached:
        break

    print()

else:
    raise TimeoutError(f'Unable to reach goal after {max_attempts} attempts')

driver.quit()