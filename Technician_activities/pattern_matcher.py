import re
import yaml
from typing import Dict

class PatternMatcher:
    """Enhanced pattern matching for ticket classification"""

    def __init__(self):
        self.patterns_db = {}
        self.load_patterns()

    def load_patterns(self):
        """Load patterns from YAML configuration."""
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
        """Save patterns to YAML file."""
        with open('patterns.yaml', 'w') as f:
            yaml.dump(self.patterns_db, f)

    def match_category(self, text: str) -> str:
        """Match text against category patterns."""
        max_score = 0
        category = 'unknown'
        
        for cat, patterns in self.patterns_db.get('categories', {}).items():
            score = sum(len(re.findall(pattern, text)) for pattern in patterns)
            if score > max_score:
                max_score = score
                category = cat
                
        return category

    def extract_environment_info(self, text: str) -> Dict[str, str]:
        """Extract system environment information from the given text."""
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

    def learn_new_pattern(self, category: str, text: str):
        """Learn new patterns from successful categorizations."""
        words = text.lower().split()
        for i in range(len(words) - 2):
            pattern = r'(?i)' + r'.{0,10}'.join(words[i:i + 3])
            if category in self.patterns_db['categories']:
                if pattern not in self.patterns_db['categories'][category]:
                    self.patterns_db['categories'][category].append(pattern)
            else:
                self.patterns_db['categories'][category] = [pattern]
        self.save_patterns()