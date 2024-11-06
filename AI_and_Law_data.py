import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the dataset
file_path = 'C:/Users/cavus/Desktop/AI and Law Paper/Megans_List.csv'  # Update with your correct file path
data = pd.read_csv(file_path)

# Convert height to numerical (inches)
def height_to_inches(height):
    try:
        if pd.isnull(height):
            return None
        feet, inches = height.split("'")
        return int(feet) * 12 + int(inches.replace('"', ''))
    except:
        return None

data['Height'] = data['Height'].apply(height_to_inches)

# Calculate age from Date of Birth
def calculate_age(dob):
    try:
        dob = datetime.strptime(dob, "%m-%d-%Y")
        today = datetime.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except:
        return None

data['Age'] = data['Date of Birth'].apply(calculate_age)

# Handle missing values in Year of Last Conviction and Year of Last Release
data['Year of Last Conviction'] = pd.to_numeric(data['Year of Last Conviction'], errors='coerce')
data['Year of Last Release'] = pd.to_numeric(data['Year of Last Release'], errors='coerce')

# Remove rows where 'Sex' is 'Unknown'
data = data[data['Sex'] != 'Unknown']

# Function to show descriptive statistics
def show_descriptive_statistics(df):
    desc_stats = df[['Age', 'Height', 'Weight', 'Year of Last Conviction', 'Year of Last Release']].describe()
    print(desc_stats)

# Function to generate figures for data visualization with 600 dpi
def generate_figures(df):
    sns.set(style="whitegrid")  # Set seaborn style for better visualisation

    # 1. Age Distribution of Offenders (colorful with 'viridis' palette)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Age'].dropna(), kde=True, bins=20, color='dodgerblue')
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    #plt.title('Age Distribution of Offenders', fontsize=16)
    plt.savefig("Age_Distribution_Offenders.png", dpi=600)
    plt.show()

    # 2. Distribution of Offenders by Ethnicity (colorful with 'Spectral' palette)
    plt.figure(figsize=(10, 6))
    sns.countplot(y=df['Ethnicity'], order=df['Ethnicity'].value_counts().index, palette='Spectral')
    plt.xlabel('Count', fontsize=14)
    plt.ylabel('Ethnicity', fontsize=14)
    #plt.title('Distribution of Offenders by Ethnicity', fontsize=16)
    plt.savefig("Distribution_Offenders_Ethnicity.png", dpi=600)
    plt.show()

    # 3. Distribution of Offenders by Sex (colorful with 'Set2' palette, excluding 'Unknown')
    plt.figure(figsize=(10, 6))
    sns.countplot(x=df['Sex'], order=df['Sex'].value_counts().index, palette='Set2')
    plt.xlabel('Sex', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    #plt.title('Distribution of Offenders by Sex (Excluding "Unknown")', fontsize=16)
    plt.savefig("Distribution_Offenders_Sex_Without_Unknown.png", dpi=600)
    plt.show()

    # 4. Correlation Heatmap for Age, Height, Weight, Year of Last Conviction, and Year of Last Release (coolwarm palette)
    plt.figure(figsize=(10, 6))
    corr_matrix = df[['Age', 'Height', 'Weight', 'Year of Last Conviction', 'Year of Last Release']].corr()
    ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', cbar=True)

    # Modify the labels for both x and y axes
    labels = ['Age', 'Height', 'Weight', 'Year of Last\nConviction', 'Year of Last\nRelease']

    ax.set_xticks([i + 0.5 for i in range(len(labels))])
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)

    ax.set_yticks([i + 0.5 for i in range(len(labels))])
    ax.set_yticklabels(labels, rotation=0, fontsize=12)

    # Adjust the layout to prevent label overlap
    plt.tight_layout()

   # plt.title('Correlation Heatmap for Age, Height, Weight, Conviction, and Release', fontsize=16)
    plt.savefig("Correlation_Heatmap_Age_Height_Conviction.png", dpi=600)
    plt.show()

# Show descriptive statistics
show_descriptive_statistics(data)

# Generate the figures with 600 dpi
generate_figures(data)

# Show descriptive statistics for all columns in the dataset
pd.set_option('display.max_columns', None)  # This ensures that all columns are shown
pd.set_option('display.expand_frame_repr', False)  # Keeps the output in one line

# Display descriptive statistics for all numerical columns
print(data.describe())
