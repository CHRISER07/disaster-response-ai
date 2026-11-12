import os

replacements = {
    '-': '-', '-': '-', ' deg ': ' deg ', ''': "'", ''': "'", '"': '"', '"': '"', 
    '...': '...', '[OK]': '[OK]', '[FAIL]': '[FAIL]', '[WARN]': '[WARN]', '[WARN]': '[WARN]', '[ALERT]': '[ALERT]',
    '[OK]': '[OK]', '[CRITICAL]': '[CRITICAL]', '[DANGER]': '[DANGER]', '[WARNING]': '[WARNING]',
    '[LOC]': '[LOC]', '[DB]': '[DB]', '[DB]': '[DB]', '[REFRESH]': '[REFRESH]', '[TEMP]': '[TEMP]',
    '[RAIN]': '[RAIN]', '[RAIN]': '[RAIN]', '[SUN]': '[SUN]', '[CHAT]': '[CHAT]', '[ALERT]': '[ALERT]',
    '[DEL]': '[DEL]', '[CLOUD]': '[CLOUD]', '[LOCAL]': '[LOCAL]', '[PLAY]': '[PLAY]', '[WARN]': '[WARN]',
    '[Y]': '[Y]', '[N]': '[N]', '>=': '>=', '[TOOL]': '[TOOL]', '[TOOL]': '[TOOL]', '[Y]': '[Y]', '[N]': '[N]'
}

modified_files = 0
for root, dirs, files in os.walk('.'):
    # Skip non-code directories
    if any(ignore in root for ignore in ['venv', '__pycache__', '.git', '.gemini', 'chroma_db', 'node_modules']):
        continue
        
    for file in files:
        if file.endswith('.py') or file.endswith('.md') or file.endswith('.txt') or file.endswith('.yml') or file.endswith('.jsonl') or file.endswith('.json') or file.endswith('.env.example') or file.endswith('Dockerfile.api') or file.endswith('Dockerfile.dashboard'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                continue
            
            original_content = content
            
            # 1. Replace common unicode with safe ASCII equivalents
            for k, v in replacements.items():
                content = content.replace(k, v)
                
            # 2. Strip ALL remaining non-ASCII characters strictly
            content = ''.join([c if ord(c) < 128 else '' for c in content])
            
            if content != original_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                modified_files += 1
                print(f"Cleaned: {filepath}")

print(f"\nDone. Removed unicode/emojis from {modified_files} files.")
