import gym
import numpy as np
import logging as log
import urllib3
import hashlib
import xxhash

from time import sleep
from os import environ
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait

from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService

from telnetlib import Telnet
from requests import Session as http_session  # noqa

log_format = '%(asctime)s | %(levelname)-8s | %(module)-5s:%(lineno)-4d | %(message)s'
file_handler = log.FileHandler('experiment.logs')
file_handler.setFormatter(log.Formatter(log_format))
console_handler = log.StreamHandler()
console_handler.setFormatter(log.Formatter(log_format))
log.basicConfig(level=log.INFO, handlers=[file_handler, console_handler])

modifiable_elements = ["input", "button", "select", "textarea", "a"]


class IODynamicEnv(gym.Env):
    def __init__(
            self, goal,
            address='http://192.168.1.1', user='admin', password='aruba123',
            console_ip='192.168.1.1', console_port=23, console_timeout=10,
            headless=False, hash_table_size=2048):

        log.info('Creating environment ...')

        # Class variables
        self.address = address
        self.user = user
        self.password = password
        self.console_ip = console_ip
        self.console_port = console_port
        self.console_timeout = console_timeout
        self.serial_number = 'CN2ZLJC041'
        self.headless = headless
        self.hash_table_size = hash_table_size

        # Environment initialization
        super(IODynamicEnv, self).__init__()
        self.action_space = None
        self.actions_list = []
        self.actions_dict = {}
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.current_config = []

        # self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Reward signal variables and functions
        self.goal = goal

        environ['WDM'] = '0'
        environ['WDM_LOG'] = '0'
        environ['WDM_SSL_VERIFY'] = '0'
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Web driver initialization
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        if self.headless:
            options.add_argument("--headless")
        self.driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()),
            options=options)
        self.wait = WebDriverWait(self.driver, 10)
        self.driver.get(self.address)

        # Login
        self._login()
        self._wait_js_queries()

        # Get initial state
        self._console_set_clean_env()
        self.current_state = self.get_state()

    def __del__(self):
        try:
            self.driver.close()
        except Exception:
            pass

        try:
            self.driver.quit()
        except Exception:
            pass

    def step(self, action):
        if action >= self.action_space.n:
            raise ValueError("Action out of bounds")

        last_state = self.current_state
        log.info(f'State: {self.get_state_str(self.current_state)}')

        # Reward logic:
        # 1. Action is unavailable (not visible or disabled).
        # - R(s) = -10

        # 2. Action generated an error (GUI showed an error).
        # - R(s) = -5

        # 3. Action didn't generate any change in the environment
        # (config file didn't change).
        # - R(s) = 0

        # 4. Action generated a change (any change).
        # - R(s) = 5

        # 5. Action generated a desired change (is this tied to the current goal?).
        # - R(s) = 10

        element_id = self.actions_list[action]
        element_info = self.actions_dict[element_id]
        element_id_type = element_info[0]
        element_type = element_info[1]

        # Find the element by type ['id', 'text', 'accessible_name', 'data-id']:
        try:
            if element_id_type == 'id':
                element = self.driver.find_element(By.ID, element_id)

            elif element_id_type == 'text':
                element = self.driver.find_element(By.LINK_TEXT, element_id)

            elif element_id_type == 'data-id':
                element = self.driver.find_element(By.XPATH, f'//*[@data-id="{element_id}"]')

            elif element_id_type == 'checkbox':
                element = self.driver.find_element(By.ID, element_id)

            else:
                log.warning(f'Element id type ({element_id_type}) is not supported for element id {element_id}.')
                raise Exception

            # Check if element is enabled
            if not element.is_enabled():
                log.warning(f'Element {element_id} is not enabled.')
                raise Exception

        except Exception:
            # Case 1: Action is not available
            log.warning(f'Unable to find element ({element_id}) required to perform the action')
            reward = -10
            done = False
            return self.current_state, reward, done, {}

        # Scroll to the required element
        self.driver.execute_script('arguments[0].scrollIntoView({block:"center", inline:"center"});', element)

        log.info(f'Performing action {action} [{element_id} - {element_type}]')

        # Perform action
        # modifiable_elements = ["input", "button", "select", "textarea", "a", "nav-link"]
        apply_required = False
        try:
            if element_type == 'input':
                self.wait.until(expected_conditions.element_to_be_clickable(element))
                element.clear()
                element.send_keys('textplaceholder')
                apply_required = True

            elif element_type == 'text':
                self.wait.until(expected_conditions.element_to_be_clickable(element))
                element.clear()
                element.send_keys('textplaceholder')
                apply_required = True

            elif element_type == 'checkbox':
                self.driver.execute_script("arguments[0].click();", element)
                apply_required = True

            elif element_type == "button":
                self.wait.until(expected_conditions.element_to_be_clickable(element))
                element.click()
                apply_required = True

            elif element_type == 'select':
                log.error(f'Unsupported element type {element_type}')
                import pdb
                pdb.set_trace()

            elif element_type == 'textarea':
                self.wait.until(expected_conditions.element_to_be_clickable(element))
                element.clear()
                element.send_keys('textplaceholder')
                apply_required = True

            elif element_type == 'a':
                self.wait.until(expected_conditions.element_to_be_clickable(element))
                element.click()

            elif 'nav-link' in element_type:
                self.wait.until(expected_conditions.element_to_be_clickable(element))
                element.click()

            else:
                log.error(f'Unsupported element type, {element_type} for element {element_id}.')
        except Exception:
            # Case 2: Perform the action is not posible
            log.warning('Unable to execute action')
            reward = -10
            done = False
            return self.current_state, reward, done, {}

        # Apply changes
        if apply_required:
            try:
                self.wait.until(
                    expected_conditions.element_to_be_clickable(
                        (By.ID, "btnApply"))).click()
            except Exception:
                log.warning('Unable to apply changes')

        # Check if the use interface generated an error
        js_code = """
            return !!document.querySelector('div.error') ||
                !!document.querySelector('span.error') ||
                !!document.querySelector('p.error') ||
                !!document.querySelector('div.alert-error');
        """
        has_error = self.driver.execute_script(js_code)
        if has_error:

            log.warning('Unable to execute action')
            try:
                self.wait.until(
                    expected_conditions.element_to_be_clickable(
                        (By.ID, "btnTopRefresh"))).click()
            except Exception:
                pass
            reward = -5
            done = False
            self.current_state = self.get_state()
            log.info(f'New state: {self.get_state_str(self.current_state)}')
            return self.current_state, reward, done, {}

        # log.info(self.current_config)

        new_config = self._console_get_config()
        added_lines = set(new_config) - set(self.current_config)
        removed_lines = set(self.current_config) - set(new_config)
        changed_lines = list(added_lines.union(removed_lines))

        self.current_config = new_config[::]
        self.current_state = self.get_state()
        log.info(f'New state: {self.get_state_str(self.current_state)}')

        if len(changed_lines) == 0:
            # Case 3: Change didn't affect the environment
            if np.array_equal(last_state, self.current_state):
                log.warning('Action did not change the environment at all')
                # if element_type == 'button':
                #     reward = -5
                # else:
                reward = 0
            else:
                log.warning('Action did not change the environment internally')
                reward = 0
        else:
            # Case 4: Change affect the environment
            log.info('Action changed the environment')
            log.info(f'Added lines: {added_lines}')
            log.info(f'Removed lines: {removed_lines}')

            if added_lines.intersection(set(self.goal)):
                log.info('New lines got the environment closer to the goal')
                reward = 5
            elif removed_lines.intersection(set(self.goal)):
                log.info('New lines got the environment farther to the goal')
                reward = -5
            else:
                log.warning('New lines did not affect the environment goal')
                reward = -1

        hits = self._get_config_hits(new_config, self.goal)

        if changed_lines != 0:
            log.debug('------------')
            log.debug(f'{len(self.goal)=}')
            log.debug(f'{hits=}')
            log.debug('new_config=')
            for line in new_config:
                log.debug(line)
            log.debug('------------')

        done = (hits == len(self.goal))

        # Get the reward and done
        log.debug(f"Action: {action} -> Reward: {reward}, Done: {done}")

        # Update the state
        return self.current_state, reward, done, {}

    def reset(self, soft=True):

        if soft:
            # Soft factory reset
            self._console_soft_reboot()
            self.driver.get(self.address)
        else:
            # Real factory reset
            self._restapi_reboot()
            self._set_mgmt_local_mode()
            self._console_set_credentials()
            for _ in range(3):
                try:
                    self._login()
                    break
                except Exception:
                    self.driver.get(self.address)
                    self.driver.refresh()
            else:
                raise Exception('Unable to login')

        self._wait_js_queries()
        self.current_config = self._console_get_config()
        self.current_state = self.get_state()
        log.info(f'State: {self.get_state_str(self.current_state)}')
        return self.current_state

    def close(self):
        self.driver.quit()
        pass

    def find_actions(self):

        new_actions = 0

        if self.action_space is None:
            sleep(0.5)

        self._wait_js_queries()

        # Step 1: Get all modifiable elements by tag name in
        # one go using JavaScript
        for _ in range(3):
            try:
                interactive_elements = self.driver.find_elements(
                    By.XPATH, "//input | //button | //a | //select | //textarea")

                # Step 2: Filter visible and unsupported classes in
                # one go using JavaScript
                unsupported_classes = [
                    'btn dropdown-toggle btn-light',
                    'page-link',
                    'navbar-toggler px-3 mr-0',
                    'filterTable',
                    'close',
                    'selectpicker',
                    'form-control timepick-input',
                    'form-control datetimepicker-input',
                    'dt-checkboxes'
                ]

                # JavaScript to filter elements and get necessary
                # attributes (class, id, text, etc.)
                js_code = """
                    return arguments[0].map(el => ({
                        element: el,
                        displayed: el.offsetParent !== null,
                        className: el.className
                    })).filter(el => el.displayed && !arguments[1].includes(el.className));
                """

                # Execute JavaScript to filter visible elements
                # and unsupported classes
                filtered_elements = self.driver.execute_script(
                    js_code, interactive_elements, unsupported_classes)
                break
            except Exception:
                log.error('Unable to find actions.')
                sleep(1)
        else:
            raise Exception('Unable to find actions.')

        # Forbidden actions, used to limit the scope
        forbidden_actions = [
            'menuCollapse',
            'gettingStartedConfBtn',
            'vlanConfBtn',
            # 'Setup Network',
            'Get Connected',
            'menuIPv4Setup',
            'menuIPv6Setup',
            'rdoIPv4AddressType_0',
            'rdoIPv4AddressType_1',
            'txtIPv4Address',
            'txtIPv4SubnetMask',
            'txtIPv4Gateway',
            'menuHttp',
            'menuHttps',
            'chkHttpState',
            'txtHttpPort',
            'txtHttpSoftTimeout',
            'txtHttpHardTimeout',
            'System Time',
            # 'User Management',
            'chkPasswordAging',
            'user-accounts-add',
            'DHCP Server',
            'Schedule Configuration',
            'DNS Configuration',
            'Stacking Configuration',

            # 'Switching',
            'chkFlowControl',
            'chkJumboFrames',
            'interface-all',
            'Port Mirroring',
            'Loop Protection',
            'IGMP Snooping',
            'SNMP',
            'Interface Auto Recovery',
            'Trunk Configuration',
            'EEE Configuration',

            'Spanning Tree',
            'VLAN',
            'Neighbor Discovery',
            'Power Over Ethernet',
            'Routing',
            'Quality of Service',
            'Security',
            'Diagnostics',
            'Maintenance',
            'btnApply',
            'btnLogs',
            'lblLogOut'
        ]

        # Step 3: Process filtered elements and build the actions list and dictionary
        new_actions = 0
        for el_data in filtered_elements:
            element = el_data['element']

            # Fetch element attributes for known actions in one go
            id_type = None
            id_value = None

            for idt in ['id', 'text', 'accessible_name', 'data-id', 'outerText']:
                try:
                    id_value = element.get_attribute(idt)
                except Exception:
                    continue
                if id_value:
                    id_type = idt
                    break

            if id_value in forbidden_actions:
                continue

            if id == 'inputUsername':
                raise Exception('UI not logged in.')

            # If no valid ID found, log errors and skip
            if not id_value:
                name = element.get_dom_attribute("name")
                clss = element.get_dom_attribute("class")
                data_id = element.get_dom_attribute("data-id")
                log.error(f'Unable to get a valid id for element {element.id}, name={name}, class={clss}, data-id={data_id}')
                continue

            # Add the element to the actions list/dictionary if it's not already there
            if id_value not in self.actions_dict:
                element_type = element.get_attribute('type') or element.get_attribute('class')
                self.actions_list.append(id_value)
                self.actions_dict[id_value] = (id_type, element_type)
                new_actions += 1
                if new_actions == 1:
                    log.info(f'Adding a new action ... {id_value} [{element_type}]')
                else:
                    log.info(f'                    ... {id_value} [{element_type}]')

        if new_actions != 0:
            log.info(f"New actions were added to the env [{new_actions}] ... ")
        else:
            log.debug("No new actions we found ...")
            return 0

        # # Logic to dynamically add a new action
        new_n = new_actions
        if self.action_space is not None:
            new_n += self.action_space.n
        self.action_space = gym.spaces.Discrete(new_n)

        return new_actions

    def hash_one_hot_encode(self, features):

        num_categories = self.hash_table_size
        one_hot_matrix = np.zeros(num_categories)

        collitions = 0
        for _, text in enumerate(features):
            # Hash the string and use modulo to get a category index
            # hash_value = int(hashlib.sha1(text.encode()).hexdigest(), 16)
            hash_value = int(xxhash.xxh32(text).hexdigest(), 16)
            category = hash_value % num_categories

            # Check if there is a category already assign
            if one_hot_matrix[category]:
                log.warning(f'Collition found [{category} - {text}].')
                collitions += 1

            # Set the appropriate element to 1
            one_hot_matrix[category] = 1

        non_zero_count = np.count_nonzero(one_hot_matrix)
        log.info(f'Hash encoding array utilization: {non_zero_count}/{num_categories}')

        if collitions != 0:
            log.warning(f'Hash encoding process got {collitions} collitions')

        return one_hot_matrix

    def get_state(self):

        self._wait_js_queries()
        sleep(0.5)

        # Get the interactive of all the visible/enabled/present elements in the page.
        interactive_elements = self.driver.find_elements(
            By.XPATH, "//input | //button | //a | //select | //textarea")
        log.debug(f'num interactive_elements: {len(interactive_elements)}')

        # Use JavaScript to get visible and enabled elements in one batch
        js_code = """
            return Array.from(arguments[0]).filter(el =>
                el.offsetParent !== null &&
                !el.disabled &&
                window.getComputedStyle(el).display !== 'none' &&
                window.getComputedStyle(el).visibility !== 'hidden'
            );
        """

        enabled_elements = self.driver.execute_script(js_code, interactive_elements)
        log.debug(f'num enabled_elements: {len(enabled_elements)}')

        # Get unique names of the elements (by tag name, id, name, or text)
        unique_elements = set()
        log.debug(f'num unique_elements: {len(unique_elements)}')

        js_code = """
            return arguments[0].map(el => ({
                tagName: el.tagName.toLowerCase(),
                id: el.id,
                name: el.name,
                text: el.textContent.trim()
            }));
        """

        # Fetch the attributes for all enabled elements in a single batch
        element_data = self.driver.execute_script(js_code, enabled_elements)

        for el in element_data:
            element_name = f"[{el['tagName']}]".strip()  # Add tag name as identifier
            if el['id']:
                element_name += f"#[{el['id']}]".strip()  # Add id if present
            if el['name']:
                element_name += f"#[{el['name']}]".strip()
            if el['text']:
                element_name += f"#[{el['text']}]".strip()  # Add text content if present
            unique_elements.add(element_name)

        # for line in unique_elements:
        #     print(line)

        # Get the lines in the config file
        config_file = self._console_get_config()
        config_file.sort()

        # Create the Raw State and encode it
        state_encodding = self.hash_one_hot_encode(
            features=list(unique_elements)+list(config_file))

        return tuple(state_encodding)

    def get_state_str(self, state_encodding):
        # Create a hash to represent state
        binary_string = ''.join(str(int(bit)) for bit in state_encodding)
        state_encodding_str = hex(int(binary_string, 2))
        # state_encodding_str_hash = hashlib.sha1(state_encodding_str.encode("utf-8")).hexdigest()
        state_encodding_str_hash = xxhash.xxh32(state_encodding_str).hexdigest()
        return state_encodding_str_hash

    def _console_get_config(self):
        tn = Telnet(self.console_ip, self.console_port, timeout=self.console_timeout)

        ret = tn.read_until(b"User Name:", timeout=self.console_timeout).decode()
        for line in ret.split('\n'):
            log.debug(line)

        tn.write(self.user.encode() + b"\n")
        ret = tn.read_until(b"Password:", timeout=self.console_timeout).decode()
        for line in ret.split('\n'):
            log.debug(line)

        tn.write(self.password.encode() + b"\n")
        ret = tn.read_until(b"#", timeout=self.console_timeout).decode()
        for line in ret.split('\n'):
            log.debug(line)

        # Get the running config
        limit = 3
        for it in range(limit):
            config_lines = []
            tn.write(b'show running-config\n')
            for _ in range(10):
                ret = tn.read_until(b"#", 1).decode()
                new_lines = ret.split('\n')
                new_lines = [item for item in new_lines if item.strip()]
                config_lines.extend(new_lines)
                if new_lines != [] and '#' in new_lines[-1]:
                    break
                tn.write(b" ")

            # Make sure we get all the lines
            if '#' not in config_lines[-1]:
                log.warning(f"Error parsing config output {it+1}/{limit}")
                for line in config_lines:
                    log.debug(line)
                continue
            break

        else:
            raise RuntimeError(f"Unable to parse config output after {limit} tries.")

        # Remove non important lines
        output = []
        for line in config_lines:
            if 'show running-config' in line:
                continue
            elif 'Connected to port' in line:
                continue
            elif 'config-file-header' in line:
                continue
            elif 'More: <space>' in line:
                continue
            elif 'unit-type' in line:
                continue
            elif 'SKU Description' in line:
                continue
            elif 'vInstantOn' in line:
                continue
            elif '!' == line.strip():
                continue
            elif '#' in line:
                continue
            elif '@' in line:
                continue
            elif line == '\r':
                continue
            else:
                output.append(line.replace('\r', '').strip())
        return output

    def _console_reboot(self):
        tn = Telnet(self.console_ip, self.console_port, timeout=self.console_timeout)

        ret = tn.read_until(b"User Name:").decode()
        for line in ret.split('\n'):
            log.debug(line)

        tn.write(self.user.encode() + b"\n")
        ret = tn.read_until(b"Password:").decode()
        for line in ret.split('\n'):
            log.debug(line)

        tn.write(self.password.encode() + b"\n")
        ret = tn.read_until(b"#").decode()
        for line in ret.split('\n'):
            log.debug(line)

        tn.write(b"reload\n")
        ret = tn.read_until(b"\n").decode()
        tn.write(b'Y')
        ret = tn.read_until(b"\n").decode()
        for line in ret.split('\n'):
            log.debug(line)

        sleep(2)

        tn.write(b'Y')
        ret = tn.read_until(b"\n").decode()
        for line in ret.split('\n'):
            log.debug(line)

        sleep(60)

        # Wait until the reboot has finished, timeout is higher here
        for i in range(30):
            ret = tn.read_until(b"User Name:", 10).decode()
            log.debug(ret)
            if "User Name:" in ret:
                break
            if i == 18:
                tn.close()
                tn = Telnet(self.console_ip, self.console_port, timeout=self.console_timeout)
        else:
            log.error('Unable to get the login prompt')
        tn.write(self.user.encode() + b"\n")

        # After a factory reset, password is empty
        ret = tn.read_until(b"Password:").decode()
        for line in ret.split('\n'):
            log.debug(line)
        tn.write(b"\n")

        ret = tn.read_until(b"Enter new username:").decode()
        for line in ret.split('\n'):
            log.debug(line)
        tn.write(self.user.encode() + b"\n")

        ret = tn.read_until(b"Enter new password:").decode()
        for line in ret.split('\n'):
            log.debug(line)
        tn.write(self.password.encode() + b"\n")

        ret = tn.read_until(b"Confirm new password:").decode()
        for line in ret.split('\n'):
            log.debug(line)
        tn.write(self.password.encode() + b"\n")

        tn.write(b"exit\n")
        ret = tn.read_until(b"\n").decode()
        for line in ret.split('\n'):
            log.debug(line)
        tn.close()

    def _console_soft_reboot(self):
        tn = Telnet(self.console_ip, self.console_port, timeout=self.console_timeout)

        ret = tn.read_until(b"User Name:").decode()
        for line in ret.split('\n'):
            log.debug(line)
        tn.write(self.user.encode() + b"\n")

        ret = tn.read_until(b"Password:").decode()
        for line in ret.split('\n'):
            log.debug(line)
        tn.write(self.password.encode() + b"\n")

        ret = tn.read_until(b"#").decode()
        for line in ret.split('\n'):
            log.debug(line)

        tn.write(b"configure\n")
        ret = tn.read_until(b"\n").decode()

        # Clear each command one by one
        tn.write(b"hostname " + self.serial_number.encode() + b"\n")
        ret = tn.read_until(b"\n").decode()

        soft_reset_cmds = [
            'no snmp-server location',
            'no snmp-server contact',
            'no clock source sntp',
            'no sntp unicast client enable',
            'no sntp unicast client poll',
            'no sntp server',
            'no sntp port'

            "passwords complexity enable"
            "no passwords lockout",
            "no passwords complexity no-repeat",

            "no storm-control broadcast",
            "no storm-control multicast",
            "no storm-control unicast",
            "link-flap prevention enable"
        ]

        for cmd in soft_reset_cmds:
            tn.write(cmd.encode() + b"\n")
            ret = tn.read_until(b"\n").decode()
            for line in ret.split('\n'):
                log.debug(line)

        tn.write(b"exit\n")
        ret = tn.read_until(b"\n").decode()
        for line in ret.split('\n'):
            log.debug(line)
        tn.close()

    def _console_set_credentials(self):
        tn = Telnet(self.console_ip, self.console_port, timeout=self.console_timeout)

        # Wait until the reboot has finished, timeout is higher here
        for i in range(30):
            ret = tn.read_until(b"User Name:", 10).decode()
            log.debug(ret)
            if "User Name:" in ret:
                break
            if i == 18:
                tn.close()
                tn = Telnet(self.console_ip, self.console_port, timeout=self.console_timeout)
        else:
            log.error('Unable to get the login prompt')
        tn.write(self.user.encode() + b"\n")

        # After a factory reset, password is empty
        ret = tn.read_until(b"Password:").decode()
        for line in ret.split('\n'):
            log.debug(line)
        tn.write(b"\n")

        ret = tn.read_until(b"Enter new username:").decode()
        for line in ret.split('\n'):
            log.debug(line)
        tn.write(self.user.encode() + b"\n")

        ret = tn.read_until(b"Enter new password:").decode()
        for line in ret.split('\n'):
            log.debug(line)
        tn.write(self.password.encode() + b"\n")

        ret = tn.read_until(b"Confirm new password:").decode()
        for line in ret.split('\n'):
            log.debug(line)
        tn.write(self.password.encode() + b"\n")

        tn.write(b"configure\n")
        ret = tn.read_until(b"\n").decode()
        for line in ret.split('\n'):
            log.debug(line)

        # Add some configuration extra to clean the output
        tn.write(b"logging console emergencies\n")
        ret = tn.read_until(b"\n").decode()
        for line in ret.split('\n'):
            log.debug(line)

        tn.write(b"ip http timeout-policy 86400\n")
        ret = tn.read_until(b"\n").decode()
        for line in ret.split('\n'):
            log.debug(line)

        tn.write(b"ip http timeout-policy absolute 604800\n")
        ret = tn.read_until(b"\n").decode()
        for line in ret.split('\n'):
            log.debug(line)

        tn.write(b"exit\n")
        ret = tn.read_until(b"\n").decode()
        for line in ret.split('\n'):
            log.debug(line)
        tn.close()

    def _console_set_clean_env(self):
        tn = Telnet(self.console_ip, self.console_port, timeout=self.console_timeout)

        ret = tn.read_until(b"User Name:").decode()
        for line in ret.split('\n'):
            log.debug(line)
        tn.write(self.user.encode() + b"\n")

        ret = tn.read_until(b"Password:").decode()
        for line in ret.split('\n'):
            log.debug(line)
        tn.write(self.password.encode() + b"\n")

        ret = tn.read_until(b"#").decode()
        for line in ret.split('\n'):
            log.debug(line)

        tn.write(b"configure\n")
        ret = tn.read_until(b"\n").decode()
        for line in ret.split('\n'):
            log.debug(line)

        tn.write(b"logging console emergencies\n")
        ret = tn.read_until(b"\n").decode()
        for line in ret.split('\n'):
            log.debug(line)

        tn.write(b"ip http timeout-policy 86400\n")
        ret = tn.read_until(b"\n").decode()
        for line in ret.split('\n'):
            log.debug(line)

        tn.write(b"ip http timeout-policy absolute 604800\n")
        ret = tn.read_until(b"\n").decode()
        for line in ret.split('\n'):
            log.debug(line)

        tn.write(b"exit\n")
        ret = tn.read_until(b"\n").decode()
        for line in ret.split('\n'):
            log.debug(line)
        tn.close()

    def _restapi_reboot(self):

        session = http_session()
        session.trust_env = False

        # Start a new session to retrieve the session ID
        response = session.get(
            f'{self.address}/System.xml?action=login&user={self.user}&password={self.password}',
            verify=False)

        # Get the session ID needed to properly send the requests
        session_id = response.headers['sessionID']

        headers = {
            'Accept': 'application/xml, text/xml, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9,es-US;q=0.8,es;q=0.7,ko;q=0.6',
            'Connection': 'keep-alive',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Cookie': f'ifFirstBannerWelcomeMessage=true; firstWelcomeBanner=true; activeLangId=english; sessionID={session_id}&; userName={self.user}',
            'DNT': '1',
            'Origin': self.address,
            'Referer': f'{self.address}/cs654fcc5c/hpe/home.htm',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'X-Requested-With': 'XMLHttpRequest',
        }

        data = '<?xml version=\'1.0\' encoding=\'utf-8\'?><DeviceConfiguration>\n<GlobalActions action="set"><factoryDefaultAndReboot></factoryDefaultAndReboot></GlobalActions>\n</DeviceConfiguration>'

        try:
            response = session.post(
                f'{self.address}/cs654fcc5c/hpe/wcd?{{SystemGlobalSetting}}',
                headers=headers,
                data=data,
                timeout=30
            )
        except Exception:
            pass

    def _set_mgmt_local_mode(self):
        session = http_session()
        session.trust_env = False

        for _ in range(30):
            try:
                response = session.get(f'{self.address}:8080', verify=False, timeout=2.5, proxies=None)
                if response.status_code == 200:
                    break
            except Exception as err:
                log.debug(err)
            sleep(10)
        else:
            log.error(f'Unable to set device on {self.address} to local mode')

        response = session.put(
            f'{self.address}:8080/api/managementPersonality',
            json={'locallyManaged': True},
            verify=False,
        )
        if response.status_code != 200:
            log.error(f'Unable to set device on {self.address} to local mode')
        return

    def _get_config_hits(self, config1, config2):
        c1 = [line1.replace('textplaceholder', '') for line1 in config1[::]]
        c2 = [line2.replace('textplaceholder', '') for line2 in config2[::]]
        return len(list(set(c1).intersection(c2)))

    def _wait_js_queries(self, repeat=60, delay=0.2, refresh_allowed=False):
        for i in range(repeat):
            is_active_ajax_request = self.driver.execute_script(
                "return ($.active);")
            if is_active_ajax_request <= 0:
                break

            if i == int(repeat * .75):
                log.warning(
                    "Waiting for JavaScript queries to complete is taking a long time.")
                if refresh_allowed:
                    current_url = self.driver.current_url
                    log.warning('Refreshing page to complete pending queries.')
                    self.driver.refresh()
                    self.driver.get(current_url)
            sleep(delay)
        else:
            log.warning(
                f'Error waiting JS queries to finish after {delay*repeat}s, '
                'this may affect test result.')

    # Helper function to log into the Web UI
    def _login(self):
        input_username = self.wait.until(
            expected_conditions.element_to_be_clickable((By.ID, "inputUsername")))
        input_username.clear()
        input_username.send_keys(self.user)

        input_password = self.wait.until(
            expected_conditions.element_to_be_clickable((By.ID, "inputPassword")))
        input_password.clear()
        input_password.send_keys(self.password)

        submit_button = self.wait.until(
            expected_conditions.element_to_be_clickable((By.ID, "submitButton")))
        submit_button.click()
        return

    # Helper function to log out from the Web UI
    def _logout(self):
        try:
            self.short_wait.until(
                expected_conditions.element_to_be_clickable(
                    (By.ID, "btnTopUser"))).click()
            self.wait.until(
                expected_conditions.element_to_be_clickable(
                    (By.ID, "lblLogOut"))).click()
        except Exception:
            log.warning('The session has already been logged out.')
            pass
