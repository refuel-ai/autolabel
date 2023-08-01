from autolabel.transforms.pdf import PDFTransform


RESUME_PDF_CONTENT = """Page 1: Functional Resume Sample 
 
John W. Smith   
2002 Front Range Way Fort Collins, CO 80525  
jwsmith@colostate.edu  
 
Career Summary 
 
Four years experience in early childhood development with a di verse background in the care of 
special needs children and adults.  
  
Adult Care Experience  
 
• Determined work placement for 150 special needs adult clients.  
• Maintained client databases and records.  
• Coordinated client contact with local health care professionals on a monthly basis.     
• Managed 25 volunteer workers.     
 
Childcare Experience  
 
• Coordinated service assignments for 20 part -time counselors and 100 client families. 
• Oversaw daily activity and outing planning for 100 clients.  
• Assisted families of special needs clients with researching financial assistance and 
healthcare. 
• Assisted teachers with managing daily classroom activities.    
• Oversaw daily and special st udent activities.     
 
Employment History  
 1999-2002  Counseling Supervisor, The Wesley Ce nter, Little Rock, Arkansas.    
1997-1999  Client Specialist, Rainbow Special Ca re Center, Little Rock, Arkansas  
1996-1997 Teacher’s Assistant, Cowell Elem entary, Conway, Arkansas     
 
Education 
 
University of Arkansas at Little Rock, Little Rock, AR  
 
• BS in Early Childhood Development (1999) 
• BA in Elementary Education (1998) 
• GPA (4.0 Scale):  Early Childhood Developm ent – 3.8, Elementary Education – 3.5, 
Overall 3.4.  
• Dean’s List, Chancellor’s List"""


def test_pdf_transform():
    # Initialize the PDFTransform class
    transform = PDFTransform(
        output_columns=["content", "num_pages"],
        file_path_column="file_path",
        page_header="Page {page_num}: ",
        page_sep="\n\n",
    )

    # Create a mock row of data
    row = {"file_path": "tests/assets/data_loading/Resume.pdf"}

    # Transform the row
    transformed_row = transform.transform(row)

    assert set(transformed_row.keys()) == set(["content", "num_pages"])
    assert isinstance(transformed_row["content"], str)
    assert isinstance(transformed_row["num_pages"], int)
    assert transformed_row["num_pages"] == 1
    assert transformed_row["content"] == RESUME_PDF_CONTENT
