import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import PyPDF2
import docx
import pytesseract 
from PIL import Image as PILImage
import io
import platform

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model="llama-3.3-70b-versatile",
    temperature=0,
    )

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert resume analyst, ATS specialist, and career coach with 10 years of experience helping candidates land high-paying jobs.."""),
    ("human", """
     Analyze this resume against the job description and provide:
     
     !. MATCH SCORE: A percentage score indicating how well the resume matches the job description.(0-100%)
     
     2. STRONG MATCHES: 3-5 skills or qualifications or experiences that align with the job description.
     
     3. MISSING KEYWORDS: Important keywords from the job description that are missing from the resume.
    
     4. SPECIFIC IMPROVEMENT SUGGESTIONS: 3-5 actionable suggestions to improve the resume's match with the job description.
     
     5. OVERALL VERDICT: One paragrah summary of fit for the job.
     
     Resume:
     {resume}
     
     Job Description:
     {job_description}
     """)
])

def extract_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_from_docx(file):
    document = docx.Document(io.BytesIO(file.read()))
    text = ""
    for para in document.paragraphs:
        text += para.text + "\n"
    return text

def extract_from_image(file):
    image = PILImage.open(file)
    image = image.convert('L')  # Convert to grayscale for better OCR results
    width, height = image.size
    image = image.resize((width * 2, height * 2))  # Resize for better OCR accuracy
    custom_config = r'--oem 3 --psm 6'  # Use LSTM OCR Engine and assume a single uniform block of text
    text = pytesseract.image_to_string(image, config=custom_config)
    return text

def extract_text(uploaded_file):
    file_type = uploaded_file.type

    if file_type.startswith("application/pdf"):
        return extract_from_pdf(uploaded_file), "pdf"
    elif file_type.startswith("application/vnd.openxmlformats"):
        return extract_from_docx(uploaded_file), "docx"
    elif file_type.startswith("image/"):
        return extract_from_image(uploaded_file), "image"
    else:
        st.error("Unsupported file type. Please upload a PDF, DOCX, or image file.")
        return None
    
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="centered")
st.title("AI Resume Analyzer 📄")
st.write("Upload your resume and paste a job description to get an instant AI powered feedback.")

st.info("Supported file formats: PDF, DOCX, and images (PNG, JPG). For best results, use a clear and well-formatted resume.")

st.divider()
uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "png", "jpg", "jpeg"],accept_multiple_files=True)

st.divider()
job_description = st.text_area("Paste the job description here", height=200, placeholder="Copy and paste the full job description from LinkedIn, Indeed, or any job board...")

st.divider()

if st.button("🔍 Analyze My Resume", use_container_width=True):
    if not uploaded_file:
        st.error("⚠️ Please upload your resume to proceed.")
    elif not job_description.strip():
        st.error("⚠️ Please paste the job description to proceed.")
    elif len(job_description.strip()) < 50:
        st.error("⚠️ Please provide a more detailed job description (at least 50 characters).")

    else:
        with st.spinner("📖 Analyzing your resume..."):
            if not uploaded_file:
                st.error("⚠️ Failed to extract text from the uploaded file. Please ensure it's a clear and well-formatted resume.")
            elif not job_description:
                st.error("⚠️ Job description is empty. Please provide a valid job description.")
            else:
                st.success("✅ Resume text extracted successfully!")

                with st.spinner("🤖 Generating analysis..."):
                    all_text = ""
                    for single_file in uploaded_file:
                        extracted, file_format = extract_text(single_file)
                        if extracted:
                            all_text += extracted + "\n"

                if len(all_text.strip()) < 50:
                    st.error("⚠️ The combined extracted resume text is too short. Please upload more detailed resumes.")
                else:
                    st.success("✅ Combined resume text is sufficient for analysis!")

                    with st.spinner("🤖 Generating analysis..."):                            
                        chain = prompt_template | llm
                        response = chain.invoke({"resume": all_text, "job_description": job_description})

                st.divider()
                st.subheader("📊 Analysis Results")
                st.write(response.content)

                st.divider()
                with st.expander("💡 View what text was extracted from yout resume"):
                    st.write(all_text)

                st.divider()
                st.info("🔍 Note: This analysis is based on the extracted text from your resume. For best results, ensure your resume is clear and well-formatted.")