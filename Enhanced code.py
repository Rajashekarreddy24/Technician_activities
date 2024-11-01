import re
import json
import cv2
import pytesseract
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
import threading
import time
import pyautogui
from queue import Queue
import threading
import time
import mouse
import keyboard
import difflib
from datetime import datetime
from typing import List
import re
import threading
from typing import Dict
import re
import yaml
import re
import json
from PIL import Image
from datetime import datetime
import difflib
import hashlib
from typing import Dict, List, Optional, Tuple, Set
import logging
from dataclasses import dataclass, asdict
import time
from queue import Queue
from datetime import datetime
import logging
import re
import logging
from datetime import datetime
from typing import Optional




@dataclass
class TicketInfo:
    ticket_id: str
    category: str
    description: str
    timestamp: datetime
    priority: str

@dataclass
class Action:
    action_type: str
    target: str
    parameters: Dict
    timestamp: datetime
    screen_location: tuple



class TicketAnalyzer:
    """Analyzes screen recordings to extract ticket information and actions"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Common ticket patterns across different systems
        self.ticket_patterns = {
            'servicedesk': r'(SD-\d+)',
            'jira': r'(TICKET-\d+)',
            'zendesk': r'#(\d+)'
        }

    def extract_ticket_info(self, frame) -> Optional[TicketInfo]:
        """Extract ticket information from a video frame using OCR"""
        try:
            # Convert frame to text using OCR
            text = pytesseract.image_to_string(frame)
            # Extract ticket ID using patterns
            ticket_id = None
            for pattern in self.ticket_patterns.values():
                match = re.search(pattern, text)
                if match:
                    ticket_id = match.group(1)
                    break
            if ticket_id:
                # Extract other relevant information
                return TicketInfo(
                    ticket_id=ticket_id,
                    category=self._extract_category(text),
                    description=self._extract_description(text),
                    timestamp=datetime.now(),
                    priority=self._extract_priority(text)
                )
            return None
        except Exception as e:
            self.logger.error(f"Error extracting ticket info: {str(e)}")
            return None

    def _extract_category(self, text: str) -> str:
        # Add logic to extract category based on your ticket system
        categories = ['Hardware', 'Software', 'Network', 'Access']
        for category in categories:
            if category.lower() in text.lower():
                return category
        return 'Uncategorized'

    def _extract_priority(self, text: str) -> str:
        priorities = ['High', 'Medium', 'Low']
        for priority in priorities:
            if priority.lower() in text.lower():
                return priority
        return 'Medium'

    def _extract_description(self, text: str) -> str:
        # Add logic to extract description
        description_pattern = r'Description:(.*?)(?=\n|$)'
        match = re.search(description_pattern, text)
        return match.group(1).strip() if match else ''




class ActionRecorder:
    """Records and analyzes technician actions"""

    def __init__(self):
        self.actions: List[Action] = []
        self.logger = logging.getLogger(__name__)

    def record_action(self, frame, previous_frame) -> Optional[Action]:
        """Detect and record actions from frame differences"""
        try:
            # Convert frames to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            # Calculate frame difference
            frame_diff = cv2.absdiff(gray_frame, gray_prev)
            # Threshold to get regions of significant change
            _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
            # Find contours of changes
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Get the largest change area
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                # Analyze the action based on location and size
                action = self._analyze_action((x, y, w, h), frame)
                if action:
                    self.actions.append(action)
                    return action
            return None
        except Exception as e:
            self.logger.error(f"Error recording action: {str(e)}")
            return None

    def _analyze_action(self, bbox: tuple, frame) -> Optional[Action]:
        """Analyze the type of action based on screen location and content"""
        x, y, w, h = bbox
        # Example action detection logic - enhance based on your needs
        if y < 100:  # Top of screen - likely menu interaction
            return Action(
                action_type="menu_click",
                target=self._get_text_at_location(frame, bbox),
                parameters={},
                timestamp=datetime.now(),
                screen_location=(x, y)
            )
        elif w > 200 and h > 100:  # Large area - might be form interaction
            return Action(
                action_type="form_interaction",
                target="form",
                parameters={"size": (w, h)},
                timestamp=datetime.now(),
                screen_location=(x, y)
            )
        return None

    def _get_text_at_location(self, frame, bbox) -> str:
        """Extract text from the specified location in the frame"""
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        return pytesseract.image_to_string(roi).strip()


class AutomationEngine:
    """Generates and executes automation scripts based on recorded actions"""

    def __init__(self):
        self.patterns: Dict[str, List[Action]] = {}
        self.logger = logging.getLogger(__name__)

    def learn_pattern(self, ticket_info: TicketInfo, actions: List[Action]):
        """Learn a new pattern from recorded actions"""
        try:
            pattern_key = f"{ticket_info.category}_{len(actions)}"
            if pattern_key not in self.patterns:
                self.patterns[pattern_key] = actions
                self.logger.info(f"Learned new pattern: {pattern_key}")
            # Save patterns to file
            self._save_patterns()
        except Exception as e:
            self.logger.error(f"Error learning pattern: {str(e)}")

    def find_matching_pattern(self, ticket_info: TicketInfo) -> Optional[List[Action]]:
        """Find a matching pattern for a given ticket"""
        try:
            # Simple pattern matching - enhance based on your needs
            pattern_key = f"{ticket_info.category}_"
            matching_patterns = [
                actions for key, actions in self.patterns.items()
                if key.startswith(pattern_key)
            ]
            if matching_patterns:
                # Return the pattern with the most actions
                return max(matching_patterns, key=len)
            return None
        except Exception as e:
            self.logger.error(f"Error finding pattern: {str(e)}")
            return None

    def execute_pattern(self, pattern: List[Action]) -> bool:
        """Execute a pattern of actions"""
        try:
            for action in pattern:
                # Add your automation execution logic here
                # This might involve pyautogui or similar libraries
                self.logger.info(f"Executing action: {action.action_type} at {action.screen_location}")
                # Add appropriate delays between actions
                # Add error handling and verification
            return True
        except Exception as e:
            self.logger.error(f"Error executing pattern: {str(e)}")
            return False

    def _save_patterns(self):
        """Save patterns to a file"""
        # Implement pattern persistence logic
        pass




class ITAutomationSystem:
    """Main class that coordinates the analyzer, recorder, and automation engine"""

    def __init__(self):
        self.analyzer = TicketAnalyzer()
        self.recorder = ActionRecorder()
        self.automation = AutomationEngine()
        self.logger = logging.getLogger(__name__)

    def process_recording(self, video_path: str):
        """Process a screen recording to learn new patterns"""
        try:
            cap = cv2.VideoCapture(video_path)
            ret, previous_frame = cap.read()
            if not ret:
                return
            ticket_info = None
            actions = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Try to extract ticket info if not already found
                if not ticket_info:
                    ticket_info = self.analyzer.extract_ticket_info(frame)
                # Record actions
                action = self.recorder.record_action(frame, previous_frame)
                if action:
                    actions.append(action)
                previous_frame = frame
            cap.release()
            # Learn the pattern if we have both ticket info and actions
            if ticket_info and actions:
                self.automation.learn_pattern(ticket_info, actions)
        except Exception as e:
            self.logger.error(f"Error processing recording: {str(e)}")

    def handle_new_ticket(self, ticket_info: TicketInfo) -> bool:
        """Handle a new ticket by finding and executing a matching pattern"""
        try:
            pattern = self.automation.find_matching_pattern(ticket_info)
            if pattern:
                return self.automation.execute_pattern(pattern)
            return False
        except Exception as e:
            self.logger.error(f"Error handling ticket: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Initialize the system
    system = ITAutomationSystem()
    # Process a recording to learn patterns
    system.process_recording("path_to_recording.mp4")
    # Handle a new ticket
    new_ticket = TicketInfo(
        ticket_id="SD-12345",
        category="Software",
        description="Password reset request",
        timestamp=datetime.now(),
        priority="Medium"
    )
    success = system.handle_new_ticket(new_ticket)
    print(f"Automation successful: {success}")




@dataclass
class TicketInfo:
    ticket_id: str
    category: str
    description: str
    timestamp: datetime
    priority: str
    submitter: str
    assigned_to: str
    status: str
    steps_to_reproduce: List[str]
    environment: Dict[str, str]

@dataclass
class ActionContext:
    window_title: str
    active_element: str
    parent_element: str
    screen_region: Tuple[int, int, int, int]

@dataclass
class Action:
    action_type: str
    target: str
    parameters: Dict
    timestamp: datetime
    screen_location: tuple
    context: ActionContext
    verification: Dict[str, str] # Expected results after action
    wait_time: float # Time to wait after action



class PatternMatcher:
    """Enhanced pattern matching for ticket classification"""

    def __init__(self):
        self.patterns_db = {}
        self.load_patterns()

    def load_patterns(self):
        """Load patterns from YAML configuration"""
        try:
            with open('patterns.yaml', 'r') as f:
                self.patterns_db = yaml.safe_load(f)
        except FileNotFoundError:
            self.patterns_db = {
                'categories': {
                    'password_reset': [
                        r'(?i)password.{0,10}reset',
                        r'(?i)forgot.{0,10}password',
                        r'(?i)change.{0,10}password'
                    ],
                    'access_request': [
                        r'(?i)request.{0,10}access',
                        r'(?i)permission.{0,10}needed',
                        r'(?i)grant.{0,10}access'
                    ],
                    'software_install': [
                        r'(?i)install.{0,10}software',
                        r'(?i)new.{0,10}application',
                        r'(?i)download.{0,10}program'
                    ]
                },
                'priorities': {
                    'high': [
                        r'(?i)urgent',
                        r'(?i)critical',
                        r'(?i)emergency'
                    ],
                    'medium': [
                        r'(?i)normal',
                        r'(?i)standard',
                        r'(?i)regular'
                    ],
                    'low': [
                        r'(?i)low',
                        r'(?i)minor',
                        r'(?i)whenever'
                    ]
                }
            }
            # Save default patterns
            self.save_patterns()

    def save_patterns(self):
        """Save patterns to YAML file"""
        with open('patterns.yaml', 'w') as f:
            yaml.dump(self.patterns_db, f)

    def match_category(self, text: str) -> str:
        """Match text against category patterns"""
        max_score = 0
        category = 'unknown'
        for cat, patterns in self.patterns_db['categories'].items():
            score = sum(len(re.findall(pattern, text)) for pattern in patterns)
            if score > max_score:
                max_score = score
                category = cat
        return category

    def learn_new_pattern(self, category: str, text: str):
        """Learn new patterns from successful categorizations"""
        words = text.lower().split()
        for i in range(len(words) - 2):
            pattern = r'(?i)' + r'.{0,10}'.join(words[i:i + 3])
            if category in self.patterns_db['categories']:
                if pattern not in self.patterns_db['categories'][category]:
                    self.patterns_db['categories'][category].append(pattern)
            else:
                self.patterns_db['categories'][category] = [pattern]
        self.save_patterns()




class EnhancedTicketAnalyzer(TicketAnalyzer):
    """Enhanced ticket analyzer with better pattern recognition"""

    def __init__(self):
        super().__init__()
        self.pattern_matcher = PatternMatcher()
        self.ocr_queue = Queue()
        self.start_ocr_worker()

    def start_ocr_worker(self):
        """Start background OCR processing"""
        def worker():
            while True:
                frame = self.ocr_queue.get()
                if frame is None:
                    break
                self._process_frame_ocr(frame)

        self.ocr_thread = threading.Thread(target=worker, daemon=True)
        self.ocr_thread.start()

    def _process_frame_ocr(self, frame):
        """Process OCR in background"""
        try:
            # Enhance image for better OCR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # OCR with improved configuration
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(thresh, config=custom_config)
            # Store result in class variable
            self.latest_ocr_result = text
        except Exception as e:
            self.logger.error(f"OCR processing error: {str(e)}")

    def extract_environment_info(self, text: str) -> Dict[str, str]:
        """Extract system environment information"""
        env_info = {}
        patterns = {
            'os': r'OS:\s*(.*?)(?:\n|$)',
            'browser': r'Browser:\s*(.*?)(?:\n|$)',
            'software_version': r'Version:\s*(.*?)(?:\n|$)'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                env_info[key] = match.group(1).strip()
        return env_info




class EnhancedActionRecorder(ActionRecorder):
    """Enhanced action recorder with more detailed action capture"""

    def __init__(self):
        super().__init__()
        self.known_elements = set()
        self.action_patterns = []
        self.last_actions: List[Action] = []

    def start_recording(self):
        """Start recording user actions"""
        self.recording = True
        self.record_thread = threading.Thread(target=self._record_actions)
        self.record_thread.start()

    def stop_recording(self):
        """Stop recording user actions"""
        self.recording = False
        self.record_thread.join()

    def _record_actions(self):
        """Record user actions in real-time"""
        while self.recording:
            # Monitor mouse clicks
            if mouse.is_pressed():
                x, y = mouse.get_position()
                self._handle_click(x, y)
            # Monitor keyboard input
            if keyboard.is_pressed('ctrl+c'):
                self._handle_keyboard_shortcut('copy')
            elif keyboard.is_pressed('ctrl+v'):
                self._handle_keyboard_shortcut('paste')
            time.sleep(0.1)

    def _handle_click(self, x: int, y: int):
        """Handle mouse click events"""
        try:
            # Capture screen region around click
            region = (max(0, x-50), max(0, y-50), min(x+50, 1920), min(y+50, 1080))
            screenshot = pyautogui.screenshot(region=region)
            # Get active window information
            window_info = self._get_window_info()
            # Create action context
            context = ActionContext(
                window_title=window_info.get('title', ''),
                active_element=self._get_element_at_position(x, y),
                parent_element=self._get_parent_element(x, y),
                screen_region=region
            )
            # Create and store action
            action = Action(
                action_type='click',
                target=f'pos_{x}_{y}',
                parameters={'x': x, 'y': y},
                timestamp=datetime.now(),
                screen_location=(x, y),
                context=context,
                verification={'element_visible': 'true'},
                wait_time=0.5
            )
            self.actions.append(action)
            self._analyze_pattern(action)
        except Exception as e:
            self.logger.error(f"Error handling click: {str(e)}")

    def _analyze_pattern(self, action: Action):
        """Analyze actions for patterns"""
        self.last_actions.append(action)
        if len(self.last_actions) > 5:
            self.last_actions.pop(0)
        # Look for repeated sequences
        self._find_repeated_sequences()

    def _find_repeated_sequences(self):
        """Find repeated sequences of actions"""
        sequence = self._serialize_actions(self.last_actions)
        for pattern in self.action_patterns:
            similarity = difflib.SequenceMatcher(None, sequence, pattern).ratio()
            if similarity > 0.8:
                self.logger.info(f"Found similar pattern with {similarity:.2f} confidence")
                return
        # Add new pattern if sequence is long enough
        if len(self.last_actions) >= 3:
            self.action_patterns.append(sequence)




class EnhancedAutomationEngine(AutomationEngine):
    """Enhanced automation engine with robust execution capabilities"""

    def __init__(self):
        super().__init__()
        self.action_handlers = {
            'click': self._handle_click,
            'type': self._handle_type,
            'shortcut': self._handle_shortcut,
            'wait': self._handle_wait
        }
        self.verification_queue = Queue()
        self.start_verification_worker()

    def start_verification_worker(self):
        """Start background verification worker"""
        def worker():
            while True:
                action = self.verification_queue.get()
                if action is None:
                    break
                self._verify_action(action)

        self.verify_thread = threading.Thread(target=worker, daemon=True)
        self.verify_thread.start()

    def execute_pattern(self, pattern: List[Action]) -> bool:
        """Execute a pattern with verification and error handling"""
        try:
            for action in pattern:
                # Check preconditions
                if not self._check_preconditions(action):
                    self.logger.warning(f"Preconditions not met for action {action.action_type}")
                    return False
                # Execute action
                handler = self.action_handlers.get(action.action_type)
                if handler:
                    success = handler(action)
                    if not success:
                        return False
                # Verify action
                self.verification_queue.put(action)
                # Wait specified time
                time.sleep(action.wait_time)
            else:
                self.logger.error(f"No handler for action type: {action.action_type}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error executing pattern: {str(e)}")
            return False

    def _check_preconditions(self, action: Action) -> bool:
        """Check if preconditions for action are met"""
        try:
            # Check if target element is visible
            if not self._is_element_visible(action.target):
                return False
            # Check if we're in the correct window
            if not self._verify_window(action.context.window_title):
                return False
            # Check if parent element is present
            if not self._verify_parent_element(action.context.parent_element):
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error checking preconditions: {str(e)}")
            return False

    def _handle_click(self, action: Action) -> bool:
        """Handle click actions"""
        try:
            x, y = action.screen_location
            # Move mouse smoothly
            pyautogui.moveTo(x, y, duration=0.5)
            # Click with error handling
            pyautogui.click(x, y)
            return True
        except Exception as e:
            self.logger.error(f"Error handling click: {str(e)}")
            return False

    def _handle_type(self, action: Action) -> bool:
        """Handle typing actions"""
        try:
            text = action.parameters.get('text', '')
            pyautogui.typewrite(text, interval=0.1)
            return True
        except Exception as e:
            self.logger.error(f"Error handling type: {str(e)}")
            return False

    def _verify_action(self, action: Action):
        """Verify action was successful"""
        try:
            # Wait for expected conditions
            max_attempts = 3
            attempt = 0
            while attempt < max_attempts:
                if self._check_verification_conditions(action):
                    return
                time.sleep(1)
                attempt += 1
            self.logger.warning(f"Action verification failed: {action.action_type}")
        except Exception as e:
            self.logger.error(f"Error verifying action: {str(e)}")

    def _check_verification_conditions(self, action: Action) -> bool:
        """Check if verification conditions are met"""
        for condition, expected in action.verification.items():
            if condition == 'element_visible':
                if not self._is_element_visible(action.target):
                    return False
            elif condition == 'text_present':
                if not self._verify_text_present(expected):
                    return False
        return True

class EnhancedITAutomationSystem:
    """Enhanced main system with better coordination and error handling"""
    
    def __init__(self):
        self.analyzer = EnhancedTicketAnalyzer()
        self.recorder = EnhancedActionRecorder()
        self.automation = EnhancedAutomationEngine()
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        self.stats = {
            'patterns_learned': 0,
            'tickets_automated': 0,
            'success_rate': 0.0,
            'failures': []
        }
        self.load_state()

    def setup_logging(self):
        """Setup enhanced logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('automation.log'),
                logging.StreamHandler()
            ]
        )

    def load_state(self):
        """Load system state from disk"""
        try:
            with open('system_state.json', 'r') as f:
                state = json.load(f)
                self.stats = state.get('stats', self.stats)
        except FileNotFoundError:
            self.logger.info("No previous state found, starting fresh")

    def save_state(self):
        """Save system state to disk"""
        try:
            state = {
                'stats': self.stats,
                'timestamp': datetime.now().isoformat()
            }
            with open('system_state.json', 'w') as f:
                json.dump(state, f)
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")

    def start_learning_mode(self, ticket_info: Optional[TicketInfo] = None):
        """Start learning mode to record new patterns"""
        try:
            if not ticket_info:
                self.logger.info("No ticket info provided, will attempt to detect from screen")
            self.recorder.start_recording()
            self.logger.info("Learning mode started")
            # Wait for user to finish demonstration
            input("Press Enter to stop recording...")
            self.recorder.stop_recording()
            self.logger.info("Learning mode stopped")
            # Process recorded actions
            actions = self.recorder.get_actions()
            if actions:
                if not ticket_info:
                    ticket_info = self.analyzer.latest_ticket_info
                # Create pattern from actions
                pattern_id = self.automation.learn_pattern(ticket_info, actions)
                self.stats['patterns_learned'] += 1
                self.save_state()
                self.logger.info(f"New pattern learned successfully: {pattern_id}")
                return pattern_id
            return None
        except Exception as e:
            self.logger.error(f"Error in learning mode: {str(e)}")
            return None

    def handle_new_ticket(self, ticket_info: TicketInfo) -> bool:
        """Handle a new ticket by finding and executing a matching pattern"""
        try:
            self.logger.info(f"Processing ticket: {ticket_info.ticket_id}")
            # Find matching pattern
            pattern = self.automation.find_matching_pattern(ticket_info)
            if not pattern:
                self.logger.info(f"No matching pattern found for ticket {ticket_info.ticket_id}")
                return False
            # Execute pattern with monitoring
            success = self._monitored_execution(pattern, ticket_info)
            # Update stats
            self._update_stats(success, ticket_info)
            return success
        except Exception as e:
            self.logger.error(f"Error handling ticket: {str(e)}")
            self._update_stats(False, ticket_info)
            return False

    def _monitored_execution(self, pattern: List[Action], ticket_info: TicketInfo) -> bool:
        """Execute pattern with monitoring and safety checks"""
        try:
            # Start monitoring
            monitor_thread = threading.Thread(
                target=self._monitor_execution,
                args=(ticket_info,)
            )
            monitor_thread.start()
            # Execute pattern
            success = self.automation.execute_pattern(pattern)
            # Stop monitoring
            self._stop_monitoring()
            monitor_thread.join()
            return success
        except Exception as e:
            self.logger.error(f"Error in monitored execution: {str(e)}")
            return False

    def _monitor_execution(self, ticket_info: TicketInfo):
        """Monitor automation execution"""
        self.monitoring = True
        while self.monitoring:
            try:
                # Check system resources
                if self._system_resources_critical():
                    self.logger.warning("System resources critical, pausing execution")
                    self.automation.pause_execution()
                # Check for unexpected windows/popups
                if self._detect_unexpected_windows():
                    self.logger.warning("Unexpected window detected")
                    self._handle_unexpected_window()
                # Verify application state
                if not self._verify_application_state():
                    self.logger.error("Invalid application state detected")
                    self.automation.abort_execution()
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in execution monitoring: {str(e)}")

    def _stop_monitoring(self):
        """Stop execution monitoring"""
        self.monitoring = False

    def _system_resources_critical(self) -> bool:
        """Check if system resources are at critical levels"""
        try:
            # Add your system resource monitoring logic here
            # Example: Check CPU, memory usage
            return False
        except Exception as e:
            self.logger.error(f"Error checking system resources: {str(e)}")
            return True

    def _detect_unexpected_windows(self) -> bool:
        """Detect unexpected windows or popups"""
        try:
            # Add your window detection logic here
            return False
        except Exception as e:
            self.logger.error(f"Error detecting windows: {str(e)}")
            return True

    def _handle_unexpected_window(self):
        """Handle unexpected windows or popups"""
        try:
            # Add your window handling logic here
            pass
        except Exception as e:
            self.logger.error(f"Error handling unexpected window: {str(e)}")

    def _verify_application_state(self) -> bool:
        """Verify application is in expected state"""
        try:
            # Add your state verification logic here
            return True
        except Exception as e:
            self.logger.error(f"Error verifying application state: {str(e)}")
            return False

    def _update_stats(self, success: bool, ticket_info: TicketInfo):
        """Update automation statistics"""
        try:
            self.stats['tickets_automated'] += 1
            if not success:
                self.stats['failures'].append({
                    'ticket_id': ticket_info.ticket_id,
                    'timestamp': datetime.now().isoformat(),
                    'category': ticket_info.category
                })
            total_tickets = self.stats['tickets_automated']
            failures = len(self.stats['failures'])
            self.stats['success_rate'] = (total_tickets - failures) / total_tickets
            self.save_state()
        except Exception as e:
            self.logger.error(f"Error updating stats: {str(e)}")

    def generate_report(self) -> Dict:
        """Generate automation performance report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats.copy(),
                'patterns': {
                    'total': self.stats['patterns_learned'],
                    'active': len(self.automation.patterns)
                },
                'performance': {
                    'success_rate': f"{self.stats['success_rate']*100:.2f}%",
                    'total_automated': self.stats['tickets_automated'],
                    'total_failures': len(self.stats['failures'])
                },
                'failure_analysis': self._analyze_failures()
            }
            return report
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return {}

    def _analyze_failures(self) -> Dict:
        """Analyze automation failures"""
        try:
            failures = self.stats['failures']
            categories = {}
            for failure in failures:
                category = failure['category']
                if category in categories:
                    categories[category] += 1
                else:
                    categories[category] = 1
            return {
                'total_failures': len(failures),
                'by_category': categories,
                'recent_failures': failures[-5:] if failures else []
            }
        except Exception as e:
            self.logger.error(f"Error analyzing failures: {str(e)}")
            return {}

def main():
    """Main function to demonstrate system usage"""
    try:
        # Initialize system
        system = EnhancedITAutomationSystem()
        # Example: Learn new pattern
        ticket_info = TicketInfo(
            ticket_id="SD-12345",
            category="password_reset",
            description="User needs password reset for application X",
            timestamp=datetime.now(),
            priority="Medium",
            submitter="john.doe@example.com",
            assigned_to="support.team@example.com",
            status="New",
            steps_to_reproduce=["User cannot login", "Password reset requested"],
            environment={"os": "Windows 10", "browser": "Chrome"}
        )
        print("Starting learning mode...")
        pattern_id = system.start_learning_mode(ticket_info)
        if pattern_id:
            print(f"Pattern learned successfully: {pattern_id}")
        # Try automated handling
        print("Testing automated handling...")
        success = system.handle_new_ticket(ticket_info)
        if success:
            print("Automation successful!")
        else:
            print("Automation failed, manual intervention required")
        # Generate report
        report = system.generate_report()
        print("\nAutomation Report:")
        print(json.dumps(report, indent=2))
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()

