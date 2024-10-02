import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
def draw_bar_chart():
    # Data for the bar chart
    heights = [0.5, 0.3, 0.2]
    labels = [
        "same ans for\nopposite state pairs",
        "diff ans for\nopposite state pairs\n& both correct",
        "diff ans for\nopposite state pairs\n& both wrong"
    ]

    # Create the figure and the axes
    plt.figure(figsize=(10, 6))
    plt.bar(labels, heights, color=['blue', 'orange', 'green'])

    # Adding labels and title
    plt.ylabel('Frequency')
    plt.title('LLM with latest prompts')

    # Rotating x-axis labels for better visibility
    plt.xticks()

    # Adjust layout to make room for the rotated x-axis labels
    plt.tight_layout()

    # Display the plot
    plt.show()

# Call the function to draw the bar chart
draw_bar_chart()
