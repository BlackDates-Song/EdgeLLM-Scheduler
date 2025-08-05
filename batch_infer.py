import csv
from infer import load_model, predict_next_state

def run_batch_inference(input_file, output_file):
    model, tokenizer, device = load_model()

    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write header
        writer.writerow(['Input', 'Prediction'])

        for row in reader:
            input_str = ','.join(row).strip()
            prediction = predict_next_state(input_str, model, tokenizer, device)
            print(f"Input: {input_str} -> Prediction: {prediction}")
            writer.writerow(row + [str(prediction)])

        print(f"Batch inference completed. Results saved to {output_file}")