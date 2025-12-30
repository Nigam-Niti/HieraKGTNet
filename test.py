netD.eval()
all_preds = []
all_targets = []
start_time = time.time()
for data, target in test_loader:
    if opt.cuda:
        data, target = data.cuda(), target.cuda()
    with torch.no_grad():
        output = netD(data)
        preds = output.max(1)[1].cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(target.cpu().numpy())
end_time = time.time()

print_and_save(f"Total inference time: {end_time-start_time:.2f} seconds")

# Generate classification report
report = classification_report(all_targets, all_preds)
print_and_save("Classification Report:\n" + report)

# Generate confusion matrix
cm = confusion_matrix(all_targets, all_preds)

# Format confusion matrix with class labels
class_labels = list(range(num_class))
df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)

# Save confusion matrix
print_and_save(f"Confusion Matrix:\n{df_cm}")
