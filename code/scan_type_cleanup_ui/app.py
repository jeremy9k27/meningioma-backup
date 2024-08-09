from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load data
output_dir = '/home/lawrence/Meningioma/data/3_SCAN_TYPE_CLEANUP'
data_file = os.path.join(output_dir, 'needs_handcheck.csv')
data_needing_handcheck = pd.read_csv(data_file).to_dict('records')
responses = []
current_index = 0

def format_text_for_html(text):
    # Replace tabs with non-breaking spaces (4 spaces per tab)
    formatted_text = text.replace('\t', '&nbsp;' * 4)
    # Replace newlines with <br> tags
    formatted_text = formatted_text.replace('\n', '<br>')
    return formatted_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    username = request.form['username'].strip()
    if not username:
        flash('Please enter your name to proceed.')
        return redirect(url_for('index'))

    return redirect(url_for('scan', username=username, index=0))

@app.route('/scan/<username>/<int:index>', methods=['GET', 'POST'])
def scan(username, index):
    global responses

    if index >= len(data_needing_handcheck):
        return redirect(url_for('finish', username=username))

    if request.method == 'POST':
        response = request.form.get('response')
        other_response = request.form.get('other_response')

        if response == 'OTHER' and other_response:
            response = other_response

        if not response or response == '----':
            flash('Please select a valid response to proceed.')
            return redirect(url_for('scan', username=username, index=index))

        item = data_needing_handcheck[index]
        responses.append({'image_id': item['id'], 'text': response})

        index += 1
        return redirect(url_for('scan', username=username, index=index))

    item = data_needing_handcheck[index]
    if isinstance(item['image_path'], str):
        image_path = item['image_path'].split('/')[-1]
    else:
        image_path = 'no-image-available.png'
    image_url = url_for('static', filename=f'thumbnails/{image_path}')
    total_scans = len(data_needing_handcheck)

    return render_template('scan.html', username=username, image_url=image_url, text=format_text_for_html(item['text']), options=['----', 'AX_2D_T2', 'AX_3D_T1_POST', 'AX_3D_T1_PRE', 'AX_ADC', 'AX_DIFFUSION', 'AX_PD', 'AX_SWI', 'AX_STIR', 'SAG_3D_FLAIR', 'SAG_3D_T2', 'DISCARD', 'OTHER'], index=index, total_scans=total_scans)

@app.route('/save_progress/<username>/<int:index>', methods=['POST'])
def save_progress(username, index):
    global responses

    # Save responses up to the current index
    progress_data = responses[:index]

    if not progress_data:
        flash("No progress to save.")
        return redirect(url_for('scan', username=username, index=index))

    # Create a filename with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/handchecked_{username}_progress_{timestamp}.csv"
    
    # Save the progress to a CSV file
    pd.DataFrame(progress_data).to_csv(filename, index=False)
    
    flash(f"Progress saved successfully to {filename}.")
    return redirect(url_for('scan', username=username, index=index))

@app.route('/finish/<username>')
def finish(username):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/handchecked_{username}_{timestamp}.csv"
    pd.DataFrame(responses).to_csv(filename, index=False)
    flash(f"Thank you for your responses! They have been saved to {filename}.")
    return render_template('finish.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
