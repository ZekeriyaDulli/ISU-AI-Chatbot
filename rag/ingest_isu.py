"""
Istinye University Knowledge Base — Automated Ingestion Script
Data & RAG Pipeline Engineer: Fares STOUHI (STU ID: 2309115179)

Run: python rag/ingest_isu.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.vector_db import VectorDBClient, DocumentIngestionPipeline

DOCUMENTS: list[dict] = [
    {
        "source": "isu_overview",
        "text": """
Istinye University (ISU) is a private, non-profit university established in 2015 in Istanbul, Turkey,
by the 21st Century Anatolian Foundation. It was founded as the academic extension of the MLP Care
Group — Turkey's largest private healthcare network, which operates Liv Hospital, Medical Park, and
VM Medical Park brands. The university operates across two primary campuses in Istanbul. The Topkapi
Campus (Maltepe Mah., Teyyareci Sami Sk., No. 3, Zeytinburnu) houses the Faculties of Medicine,
Dentistry, Pharmacy, and Health Sciences. The Vadi Istanbul Campus (Ayazağa Mah. Azerbaycan Cad.,
Vadi İstanbul 4A Blok No:3H, Sarıyer, 34396) hosts the Faculties of Engineering and Natural Sciences,
Fine Arts Design and Architecture, Economics Administrative and Social Sciences, Communication, and
Humanities and Social Sciences. A third satellite location operates in Kağıthane. Instruction is offered
in both English and Turkish across nine undergraduate faculties, one graduate institute, and two
vocational schools. The university enrolls approximately 15,000 students in total, including roughly
3,900 international students from 96 countries. ISU is ranked in the 1001–1200 global band by Times
Higher Education World University Rankings 2026 and holds the 34th position among all Turkish
universities by Webometrics. It is accredited by the Turkish Higher Education Council (YÖK).
Main switchboard: 0850 283 60 00. General email: info@istinye.edu.tr. Website: www.istinye.edu.tr.
        """,
    },
    {
        "source": "isu_cs_programs",
        "text": """
Istinye University's Faculty of Engineering and Natural Sciences is located at the Vadi Istanbul
Campus and offers Computer Engineering, Software Engineering, Civil Engineering, Electrical and
Electronics Engineering, Biomedical Engineering, Industrial Engineering, Mechanical Engineering, and
Genetics and Bioengineering. Both Computer Engineering and Software Engineering are available in
English-medium and Turkish-medium tracks. The Computer Engineering bachelor's program is a four-year
(eight-semester) degree. The curriculum covers mathematics, physics, digital logic design, computer
architecture, operating systems, computer networking, microprocessor system design, embedded systems,
machine learning, database management systems, and Python and C programming. Students must complete
two capstone project courses in the seventh and eighth semesters and fulfill at least one summer
internship. The standard undergraduate graduation requirement is completion of all program courses
totaling 240 ECTS credits with a minimum cumulative GPA of 2.00 on a 4.00 scale. Many engineering
programs are open to Erasmus exchange students. Tuition fee for Computer Engineering and Software
Engineering is approximately $8,000 per year for the 2026-2027 academic year.
        """,
    },
    {
        "source": "isu_academic_calendar",
        "text": """
Istinye University follows the European Credit Transfer and Accumulation System (ECTS) and structures
its academic year into two main semesters plus an optional summer term. For the 2025–2026 academic
year: the Fall Semester runs from October 6, 2025 through January 9, 2026. Course registration was
held September 29 – October 3, 2025. Add/drop period: October 6–18. Midterm examinations:
November 8–16, 2025. Final exams: January 10–18, 2026. The Spring Semester begins February 16, 2026
and ends May 23, 2026, with midterms March 28 – April 5 and finals June 2–11. A Summer Term runs
July 6 – August 21, 2026. The grading system uses a 4.00 scale: AA (90–100%, 4.0), BA (85–89%,
3.5), BB (80–84%, 3.0), CB (75–79%, 2.5), CC (70–74%, 2.0), DC (65–69%, 1.5), DD (60–64%, 1.0),
FD (50–59%, 0.5), FF (below 50%, 0.0), DZ (fail due to non-attendance, no exam access). A minimum
grade of CC (2.0) is required to pass a course. Students who fail may apply for a make-up exam
(mazeret sınavı) by submitting a Make-up Exam Application Form within the period announced by their
department. A Grade Improvement Exam is also available for students wishing to raise their GPA.
        """,
    },
    {
        "source": "isu_course_registration",
        "text": """
Istinye University uses the OIS (Öğrenci Bilgi Sistemi / Student Information System) portal,
accessible at https://ois.istinye.edu.tr. The e-learning platform is Blackboard Learn at
https://istinye.blackboard.com. Login credentials use your student number as username and the
password sent to your registration email. If you forget your password, click "Forgot my password"
on the OIS login page or email kayitisleri@istinye.edu.tr. First-year students have mandatory
courses assigned automatically by OIS; only elective courses require manual selection. To register:
log in to OIS, go to the Course Selection (Ders Seçme) tab, and select courses during your assigned
time window. An academic advisor must be assigned in the system before course selection can proceed —
if no advisor appears, contact Student Affairs. Maximum credit load is 42 ECTS credits per semester,
with an annual cap of 80 ECTS. Students with a GPA of 3.70 or above may apply for special approval
to exceed this. No course from the first two semesters and no already-failed course may be dropped
during add/drop. Courses registered via OIS appear automatically in Blackboard by the next day.
Financial clearance is required to complete registration; contact ogrencimuhasebe@istinye.edu.tr for
payment-related registration blocks.
        """,
    },
    {
        "source": "isu_student_id",
        "text": """
The student ID card at Istinye University is issued at the registration area immediately upon
completing your initial enrollment — no separate application is needed. It is handed to you as part
of the registration packet. If your card is lost, go to the Correspondence Department located on the
3rd Floor of the Vadi Campus and pay the replacement fee (50 TL in 2023-2024; verify the current
amount at the window). You will receive a lost card form and a payment receipt. Take both documents
to the Information Technologies (IT) Department to receive your replacement card. The student ID is
used for: entry to campus buildings and facilities, library borrowing (mandatory, sharing prohibited),
access to the gym and sports center (payment via OIS at ois.istinye.edu.tr/fi/eodeme, receipt sent
to ogrencimerkezi@istinye.edu.tr), and discounted İstanbulKart public transit. To get a discounted
transit card, bring your student ID, a passport-sized photo, a student enrollment certificate, and
the card fee to a PTT or İstanbulKart office.
        """,
    },
    {
        "source": "isu_exam_rules",
        "text": """
At Istinye University, students must bring their valid ISU student ID card to all exams for identity
verification. Mobile phones must be silent during exams; food and drink beyond water are typically
prohibited. Academic integrity is strictly enforced. Students who cheat, attempt to cheat, or
provide cheating materials receive a grade of zero (0) for that exam, and the case is referred to
the university's disciplinary committee for proceedings under the Student Discipline Regulation.
The grading scale is: AA=4.00, BA=3.50, BB=3.00, CB=2.50, CC=2.00, DC=1.50, DD=1.00, FF=0.00.
DZ means failure due to non-attendance (student is barred from the final exam). Students who miss
an exam due to a valid documented excuse (illness, bereavement, etc.) must submit a Make-up Exam
Application Form to their department within the announced deadline. Medical excuses require official
hospital documentation. The Grade Improvement Exam allows students to retake passed exams to raise
their GPA. The Single Course Exam (tek ders sınavı) is available to students who have completed all
graduation requirements except one course. All forms are available at istinye.edu.tr/en/student-registration/forms
and submitted to the Student Registration Affairs Directorate (ÖKİD).
        """,
    },
    {
        "source": "isu_attendance_policy",
        "text": """
Istinye University enforces minimum attendance thresholds for all courses. For theoretical and
lecture courses, students must attend at least 70% of classes. For practical, laboratory, and
applied courses, the minimum is 80% attendance. Attendance is tracked by the course instructor and
recorded in the OIS system; students can view their records through OIS (may lag 1-2 days behind
actual records). A student who exceeds the absence limit is recorded as DZ (Devamsız/Absent Fail),
meaning they failed the course due to non-attendance and are barred from sitting the final exam.
The DZ course must be retaken in a subsequent semester. For a medical excuse, an official doctor's
report from an accredited healthcare facility must be submitted to the department office (not to
SKS) within a reasonable timeframe after the absence. Accepted medical reports do not automatically
restore attendance hours — the percentage is calculated on the full semester. Students with prolonged
illness should contact their department's academic secretary immediately. A Registration Freeze
(Enrollment Freeze Form, available on the forms page) is available for serious situations requiring
a full semester leave.
        """,
    },
    {
        "source": "isu_documents_transcripts",
        "text": """
Istinye University students can request official documents through OIS (https://ois.istinye.edu.tr)
via the Online Document Request feature — click "Request A New Document." Available documents include:
Transcript (transkript), Student Certificate (öğrenci belgesi), Enrollment Certificate, Temporary
Graduation Certificate, Diploma Supplement, and Military Service-related documents. Electronically
signed (e-imzalı) documents obtained through OIS are immediately available for download and are
legally valid. For physical or sealed documents, visit or contact the Student Registration Affairs
Directorate (ÖKİD / Öğrenci Kayıt İşleri Direktörlüğü) at the Vadi Campus. Processing time for
physical documents is typically 1-2 business days. Standard transcripts and student certificates
for currently enrolled students have no listed fee; verify with ÖKİD for apostille or notarized
versions. Contact: kayitisleri@istinye.edu.tr | Phone: 0850 283 60 00.
        """,
    },
    {
        "source": "isu_library",
        "text": """
Istinye University has four library locations. The Vadi Central Library (main campus, 2nd floor and
H Block 6th floor) covers 2,300 m² with 375 seats, 4 group study rooms, a super-quiet zone, and
computer areas — open 24/7 on designated floors; main service hours weekdays 08:30–20:00. The
Topkapı Campus Library (810 m², 186 seats, 2 group study rooms) has a 24/7 ground floor; main
hours weekdays 08:30–17:30. The collection includes approximately 43,000 printed books and 200,000+
electronic resources across 52 licensed databases, accessible off-campus via OpenAthens using
university credentials. Borrowing limits: undergraduate students may borrow 3 books for 15 days
with up to 3 renewals; master's students 5 books for 30 days; academic staff 10 books for 30 days.
Periodicals cannot be borrowed. Renewals must be done online. Late fines: 1 Turkish Lira per book
per overdue day; fines cannot be cancelled and must be paid by bank transfer to Fibabank/Topkapı
account, then send receipt to kutuphane@istinye.edu.tr to restore borrowing privileges. Students
with unpaid fines lose borrowing access. ISU student ID is mandatory for all borrowing.
Library contact: kutuphane@istinye.edu.tr
        """,
    },
    {
        "source": "isu_transportation_shuttle",
        "text": """
Istinye University operates free shuttle services between campuses and from key Istanbul locations.
Inter-campus shuttles run between Topkapı and Vadi Istanbul campuses. From Topkapı to Vadi departures:
07:00, 07:30, 08:00, 08:30, 09:00, 10:00, 11:00, 12:00, 13:30, 14:00, 15:00. From Vadi to Topkapı:
11:30, 12:00, 12:30, 13:00, 14:00, 15:00, 16:00, 16:30, 17:30, 19:00. Travel time is approximately
30–75 minutes depending on traffic. City shuttles depart from Seyrantepe Metro Station (M2 line),
Mecidiyeköy/Trump Towers area, and Kabataş toward the Vadi Campus. Shuttles are free for all ISU
ID holders. No prior registration is required. Updated schedules at istinye.edu.tr/tr/iletisim/servis-saatleri.
By public transit: Vadi Campus is nearest to Seyrantepe M2 Metro station inside the Vadi Istanbul
complex; bus lines 47L and 41Y stop at the Vadistanbul stop. Topkapı Campus (Zeytinburnu) is served
by buses 33B, 41AT, 50B, 76D, 82, 92B, 97; metro M1A and Metrobus 34G reach Zeytinburnu station;
tram T1 also covers Zeytinburnu. Students can obtain a discounted İstanbulKart by presenting a
student ID, enrollment certificate, passport photo, and card fee at a PTT or İstanbulKart outlet.
        """,
    },
    {
        "source": "isu_international_students_residence_permit",
        "text": """
International students at Istinye University must obtain a student residence permit (ikamet izni)
after arriving in Turkey. The process is managed through the university's Residence Permit Unit.
Portal: https://ikamet.istinye.edu.tr. Office location: Vadi Istanbul D Campus, Hamidiye Mah.
Selçuklu Caddesi No:10, D Blok, Kağıthane/İstanbul. Office hours: Tuesday 09:00–15:00, Thursday
09:00–17:00. Steps: (1) Create an appointment at https://e-ikamet.goc.gov.tr; (2) Attend the
appointment at the Migration Management Directorate and give fingerprints; (3) Submit all required
documents to the university International Office within 20 days of creating the application form
and within 4 days of the fingerprint appointment. Required documents: signed application form,
passport copy + latest entry stamp, 4 biometric photos, student certificate from OIS, transcript
(for 2nd year and above), health insurance, proof of residence address (rental contract, property
deed, dormitory letter, or hotel receipt), residence permit fee receipt, and UETS document for
extensions. Passport must remain valid at least 60 days beyond the study period end date.
Contact: ikamet@istinye.edu.tr | Emergency international line: +90 537 227 93 27, +90 535 461 74 09.
        """,
    },
    {
        "source": "isu_erasmus_exchange",
        "text": """
Istinye University's Erasmus Office is located at Vadi Istanbul D Campus, 1st Floor, Hamidiye Mah.
Selçuklu Caddesi No:10 D Blok, Kağıthane/İstanbul. Contact: erasmus@istinye.edu.tr | Phone:
0850 283 60 00 | Hours: weekdays 08:30–17:15. Erasmus+ announcements are published every November;
students apply via https://turnaportal.ua.gov.tr. Eligibility for outgoing students: undergraduate
students need a minimum GPA of 2.20/4.00; master's and PhD students need 2.50/4.00. Language
proficiency at minimum B2 level is required, either via ISU's own Erasmus language exam or an
external certificate: TOEFL iBT 72+, IELTS 6.0+, Cambridge B1+, or TOEIC 750+ (certificates valid
2 years). Mobility duration: minimum 2 months or 1 semester; maximum 12 months continuously.
Monthly Erasmus grants are awarded by country group; financially disadvantaged students receive an
additional €250/month supplement. Erasmus Internship mobility (minimum 2 months abroad) does not
require a partnership agreement with the host institution. Incoming Erasmus students apply through
their home university's Erasmus office. Full list of partner universities:
erasmus.istinye.edu.tr/en/partner-universities.
        """,
    },
    {
        "source": "isu_double_major_minor",
        "text": """
Istinye University offers Double Major and Minor programs. For a Double Major, the minimum
cumulative GPA is 3.00/4.00 at the time of application (3.15/4.00 for Pharmacy). Students must
also rank in the top 20% of their class in their main program (top 10% for Pharmacy). Students not
in the top 20% may still apply if they meet at least the minimum university entrance score for the
second program. Application window is from the beginning of the 3rd semester to the beginning of
the 5th semester for 4-year bachelor's programs. The continuation GPA requirement is 2.72/4.00;
a student may drop to 2.50 once before dismissal from the double major. Successfully completing
both programs results in a second diploma. For a Minor, the minimum GPA is 2.50/4.00, application
window is from the beginning of the 3rd semester to the end of the 6th semester, and the
continuation GPA is 2.29/4.00. Students may take up to 18 additional ECTS per semester for minor
courses. ISU offers over 80 minor program options across all faculties including Data Science,
Artificial Intelligence, and Sustainability tracks. Applications are submitted through OIS during
the announced period — email applications are not accepted.
        """,
    },
    {
        "source": "isu_internship",
        "text": """
Internship requirements at Istinye University vary by department. Pharmacy students must complete
120 mandatory working days of internship to receive their diploma. Engineering and Natural Sciences
students must complete two mandatory internships of 20 working days each (40 days total). Interior
Architecture and Environmental Design students must complete 20 working days during periods
designated by the department. Health Services Vocational School (SHMYO) students have a mandatory
20-day internship after the first year. The process is managed by the Career Center (Kariyer
Merkezi). Steps: (1) Find a placement using your own resources or Career Center listings; (2) Submit
application via OIS at least 15 days before the internship start date; (3) Obtain company stamp and
wet signature on the Internship Application Form; (4) Upload signed PDF back to OIS; (5) Get
academic approvals online from Department Chair and Faculty Dean. The university covers mandatory
internship insurance premiums via SGK 4A Employment Entry Statement. Required documents: Internship
Application Form (wet signature + company stamp), SPAS Eligibility Certificate from e-Government
(e-devlet). Career Center contact: ogrencimerkezi@istinye.edu.tr | 0850 283 60 00.
        """,
    },
    {
        "source": "isu_health_psychological_services",
        "text": """
Istinye University maintains infirmaries on each campus. At the Vadi Main Campus, the infirmary is
at Entrance Floor Room Z09 (phone extension 0850 283 64 25). At Vadi H Block, it is at 2nd Floor
Room 209 (extension 0850 283 64 40). The Topkapı Campus has on-site access to Liv Hospital Topkapı
for emergency care. Infirmary hours are weekdays 08:30–17:15. Services include first aid, treatment
of sudden illness, minor injuries, low blood pressure, dizziness, and fever; basic medications are
stocked on site. Emergency cases are transferred by ambulance. The Psychological Counseling Unit
(Psikolojik Danışma Birimi, PCU) is free for all students and academic/administrative staff.
Services include individual psychotherapy sessions (~45 minutes each) and workshops on mental health
topics. To apply: complete the PCU application form and email it to pdb@istinye.edu.tr using your
university email (@stu.istinye.edu.tr). You will be contacted to schedule an appointment. PCU Lead:
Clinical Psychologist Doğa Küçük. Contact: pdb@istinye.edu.tr | 0850 283 60 00.
        """,
    },
    {
        "source": "isu_dining_campus_life",
        "text": """
Istinye University provides dining facilities on both campuses. The Topkapı Campus has 4 cafes and
restaurants including Starbucks and EspressoLab, plus a dining hall on the basement (-1) floor
serving 3 different meal options daily. Vending machines throughout the campus dispense snacks and
drinks at all times. At the Vadi Campus, multiple eating options are available within the Vadi
Istanbul complex, which is a commercial shopping mall housing the university, including the
university cafeteria and private food establishments. Monthly menus are prepared by faculty from
the Department of Nutrition and Dietetics and overseen by a food engineer through the SKS
department. Sports facilities include multi-purpose outdoor fields, an indoor sports hall, the ISU
Sport Center gym, and free table tennis areas at both campuses. The university organizes ISUCup
sports tournaments. The university supports over 80 student clubs spanning academic, cultural,
artistic, sports, gastronomy, and outdoor categories. ISU does not operate its own dormitories, but
has formal partnerships with four private student housing providers: Aren, Al-Firouz, Comfortist,
and Cumhuriyet, located in Zeytinburnu, Bayrampaşa, Vadistanbul, and Fındıkzade neighborhoods.
Contact SKS: Room Z06, Vadi Main Campus | info@istinye.edu.tr | 0850 283 60 00.
        """,
    },
    {
        "source": "isu_it_wifi_systems",
        "text": """
At Istinye University, your student email address is in the format studentnumber@stu.istinye.edu.tr.
OIS login credentials (student number + password) and email account details are sent to the personal
email you provided during registration. To log in to OIS: go to https://ois.istinye.edu.tr, enter
your student number and password; first login requires SMS phone verification. If you forget your
password, click "Forgot my password" on the OIS page or email kayitisleri@istinye.edu.tr. Blackboard
Learn is at https://istinye.blackboard.com — use the same student number and password as OIS. Courses
registered in OIS automatically appear in Blackboard by the next day. Both campuses are fully covered
by campus-wide wireless internet. Library online databases are accessible off-campus via OpenAthens
using your university credentials. ISU has official iOS and Android mobile apps for accessing
university services. For technical issues: bstdestek@istinye.edu.tr. For account and password
issues: kayitisleri@istinye.edu.tr. General IT support phone: 0850 283 60 00.
        """,
    },
    {
        "source": "isu_tuition_payment",
        "text": """
Istinye University tuition fees for 2026–2027 (in USD): Medicine $21,850–$23,000/year (Turkish) or
$27,550–$29,000 (English); Dentistry $19,000–$20,000 (Turkish) or $23,275–$24,500 (English);
Pharmacy $13,300–$14,000 (Turkish) or $14,250–$15,000 (English); Computer Engineering and Software
Engineering approximately $8,000/year; most Economics, Communication and Humanities programs
$7,000–$8,000/year. International students must pay a minimum of 50% at registration; remaining
balance due by December 1. Payment options: (1) Online Virtual POS via https://ois.istinye.edu.tr/fi/eodeme
— up to 8 installments with 3D Secure; (2) Bank Transfer/EFT to Yapı Kredi Bank, IBAN:
TR60 0006 7010 0000 0048 5290 88 — include full name, department, and ID/passport number in
description; (3) In-person credit card at the Student Accounting Office, Vadi Campus Room ANK1B07;
(4) Yapı Kredi Bank overdraft account (KMH) — 1 down payment + 7 installments, interest-free if
paid on time. Available discounts: 5% sibling discount (two siblings enrolled simultaneously), 7%
for full annual upfront payment, 10% for ISU staff and MLP Care Group employees and their immediate
family. Maximum combined discount is 15%. Late installment payments incur maturity difference
(vade farkı) charges. Contact: ogrencimuhasebe@istinye.edu.tr.
        """,
    },
    {
        "source": "isu_clubs_buddy_program",
        "text": """
Istinye University has approximately 80-100 student clubs spanning academic, arts, sports, social
responsibility, gastronomy, mountaineering, and hobby categories. Clubs are managed by the ÖMER
(Student Center / Öğrenci Merkezi) unit under the SKS department. To browse clubs: visit
istinye.edu.tr/en/sks/list-clubs — each club page lists the president's name, email, description,
and social media. To join, contact the club president directly via email or Instagram. To found a
new club: gather a minimum of 10 students, ensure the proposed club does not duplicate an existing
one, and submit a founding application using the Club Forms at istinye.edu.tr/en/sks/forms.
For new and international students, the Buddy Program (guide@istinye.edu.tr) pairs you with an
experienced upper-year student for 6 weeks of guided orientation covering campus life, Istanbul
orientation, and academic culture. Successful buddies receive rewards including Manifesto course
exemptions. Part-time job opportunities for students are also posted and managed through the Student
Center. SKS contact for all student life matters: Room Z06, Vadi Main Campus |
ogrencimerkezi@istinye.edu.tr | 0850 283 60 00.
        """,
    },
    {
        "source": "isu_graduation_requirements",
        "text": """
To graduate from Istinye University, students must: (1) achieve a minimum cumulative GPA of
2.00/4.00; (2) successfully complete all courses, practical applications, and internships specified
in the program curriculum; (3) complete the required total ECTS credits — 240 ECTS for a 4-year
(8-semester) bachelor's degree, 120 ECTS for a 2-year associate program. Students with a GPA below
2.00 despite passing all courses must retake at least one DC or DD grade course to raise the
average. Honors at graduation: Honors (Onur) for GPA 3.00–3.49; High Honors (Yüksek Onur) for GPA
3.50 and above. A Diploma Supplement is issued with every degree. Diploma types: Ön Lisans
(associate) after 4 semesters; Lisans (bachelor's) after 8 semesters (4 years); Medicine, Dentistry,
and Pharmacy have 6-year programs with additional clinical requirements. The graduation ceremony is
held annually in June at the Turkcell Vadi Open-Air Stage within the Vadi Istanbul complex.
To apply for graduation, submit the Graduation Form through OIS. The Student Registration Affairs
Directorate (ÖKİD) processes temporary graduation certificates and final diplomas.
Contact: kayitisleri@istinye.edu.tr | 0850 283 60 00.
        """,
    },
    {
        "source": "isu_admission_requirements",
        "text": """
Turkish citizens apply to Istinye University through the ÖSYM national system using YKS exam scores.
Fine Arts and Architecture programs additionally require a special ability/talent exam. International
applicants must meet the following thresholds: minimum high school success rate of 70% for Medicine
and Dentistry; 50% for all other programs. Accepted international qualification exams: SAT I
(minimum 1200 for Medicine/Dentistry/Pharmacy; 1000 for others), ACT (24/36 for health sciences;
21/36 for others), GCE A Levels (Grade A in science subjects for health programs; Grade D in two
courses for others), IB (30/45 or 28/45), and TR-YÖS (75/100 or 60/100). Language requirements:
TOEFL iBT 60+ for English-medium bachelor's programs; TOEFL iBT 75+ for master's programs.
Turkish-medium programs require at least B2-level Turkish (C1 for master's). Applicants without a
recognized language certificate must sit ISU's own English Proficiency Test; those who do not pass
must complete a one-year preparatory English program before commencing their degree. Required
documents: translated and notarized high school diploma and transcript, passport copy, passport
photograph, exam result (if applicable), and language certificate. Application window for
2025-2026: March 28 – October 31, 2025. Registration: July 1 – October 31, 2025.
Admissions contact: tanitim@istinye.edu.tr | WhatsApp: 0530 281 8888.
        """,
    },
    {
        "source": "isu_contacts_directory",
        "text": """
Istinye University complete contact directory. Main switchboard: 0850 283 60 00 (toll-free within
Turkey). WhatsApp admissions hotline: 0530 281 8888. General email: info@istinye.edu.tr.
Website: www.istinye.edu.tr. Key departmental contacts: Admissions/Prospective Students:
tanitim@istinye.edu.tr. Student Registration Affairs (ÖKİD): kayitisleri@istinye.edu.tr.
Student Accounting (tuition payments): ogrencimuhasebe@istinye.edu.tr. Erasmus Office:
erasmus@istinye.edu.tr. Residence Permit Unit: ikamet@istinye.edu.tr. Library:
kutuphane@istinye.edu.tr. Psychological Counseling (PCU): pdb@istinye.edu.tr.
IT Support (technical issues): bstdestek@istinye.edu.tr. Student Center / Gym / Clubs:
ogrencimerkezi@istinye.edu.tr. Buddy Program: guide@istinye.edu.tr.
International Office Emergency Lines: +90 537 227 93 27 and +90 535 461 74 09.
Administrative leadership: Secretary General Atilla VARLI (atilla.varli@istinye.edu.tr).
Vadi Campus address: Ayazağa Mah. Azerbaycan Cad. (Vadi İstanbul 4A) Blok No:3H, 34396
Sarıyer/İSTANBUL. Topkapı Campus address: Maltepe Mah., Teyyareci Sami Sk., No.3, 34010
Zeytinburnu/İSTANBUL. Kağıthane/International Office: Hamidiye Mah. Selçuklu Caddesi No:10
D Blok, Kağıthane/İstanbul. Faculty contact: all faculties reachable via main switchboard
0850 283 60 00 including Medicine, Dentistry, Pharmacy, Health Sciences, Nursing, Engineering,
Economics, Fine Arts, Communication, and Humanities.
        """,
    },
]


def main():
    print("Connecting to ChromaDB...")
    db = VectorDBClient()
    pipeline = DocumentIngestionPipeline(db, chunk_size=200, overlap=30)

    existing = db.count
    if existing > 0:
        print(f"Warning: database already has {existing} chunks.")
        answer = input("Clear and rebuild from scratch? (yes/no): ").strip().lower()
        if answer == "yes":
            db.delete_collection()
            db = VectorDBClient()
            pipeline = DocumentIngestionPipeline(db, chunk_size=200, overlap=30)
            print("Collection cleared.")
        else:
            print("Appending to existing database.")

    print(f"\nIngesting {len(DOCUMENTS)} documents...\n")
    total_chunks = 0
    for doc in DOCUMENTS:
        chunks_added = pipeline.ingest_text(doc["text"].strip(), source=doc["source"])
        total_chunks += chunks_added
        print(f"  ✓  {doc['source']:<45} → {chunks_added} chunk(s)")

    print(f"\nDone. Total chunks in database: {db.count}")
    print("\nSample retrieval test:")
    results = db.query("What is the tuition fee for Computer Engineering?", n_results=2)
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r[:120]}...")


if __name__ == "__main__":
    main()
