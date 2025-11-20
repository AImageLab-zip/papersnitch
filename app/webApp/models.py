from django.db import models


class Operations(models.Model):
    """Different operations of analizer"""

    name = models.CharField(max_length=100, unique=True, verbose_name="Operation type")

    class Meta:
        verbose_name = "Operations"
        verbose_name_plural = "Operations"

    def __str__(self):
        return self.name


class Conference(models.Model):

    name = models.CharField(max_length=300, verbose_name="Conference name")
    year = models.IntegerField(verbose_name="Year", blank=True, null=True)
    url = models.URLField(verbose_name="Website", blank=True, max_length=500)
    last_update = models.DateTimeField(auto_now=True, verbose_name="Last update")

    class Meta:
        ordering = ["name"]
        verbose_name = "Conference"
        verbose_name_plural = "Conferences"

    def __str__(self):
        return f"{self.name}{self.year}"


# class Author(models.Model):

#     email = models.CharField(max_length=500, verbose_name="Authot mail", blank=True, null=True)
#     name = models.CharField(max_length=500, verbose_name="Author name")
#     is_sus = models.BooleanField(default=False)

#     class Meta:
#         verbose_name = "Author"
#         verbose_name_plural = "Authors"

#     def __str__(self):
#         return self.name

# TODO aggiungere un modello relativo agli score, sia per aver salvato il valore che il testo che ha portato a tale valore


class Paper(models.Model):

    # TODO add a field related to the state of a paper (not reviewed, reviewed, review and rebuttal, published, etc)
    title = models.CharField(max_length=500, verbose_name="Title")
    doi = models.CharField(
        max_length=255, unique=True, verbose_name="DOI", blank=True, null=True
    )
    abstract = models.TextField(verbose_name="Abstract")
    supp_materials = models.FileField(
        upload_to="supp_materials",
        verbose_name="Supplementary materials",
        blank=True,
        null=True,
    )
    paper_url = models.URLField(verbose_name="Paper URL", blank=True, max_length=500)
    pdf_url = models.URLField(verbose_name="PDF URL", blank=True, max_length=500)
    code_url = models.URLField(verbose_name="Code URL", blank=True, max_length=500)
    authors = models.CharField(
        max_length=255, unique=True, verbose_name="Authors", blank=True, null=True
    )
    conference = models.ForeignKey(
        Conference,
        on_delete=models.PROTECT,
        related_name="papers",
        verbose_name="Conference",
        blank=True,
        null=True,
    )
    pdf_file = models.FileField(
        upload_to="pdf",
        blank=True,
        null=True,
    )

    reviews = models.TextField(verbose_name="All reviews text", blank=True, null=True)
    author_feedback = models.TextField(
        verbose_name="Author feedback", blank=True, null=True
    )
    meta_review = models.TextField(
        verbose_name="All Meta-reviews text", blank=True, null=True
    )
    last_update = models.DateTimeField(
        auto_now=True, verbose_name="Last update", blank=True, null=True
    )

    class Meta:
        verbose_name = "Paper"
        verbose_name_plural = "Papers"

    def __str__(self):
        return self.title


class Dataset(models.Model):

    name = models.CharField(max_length=300, verbose_name="Dataset name")
    description = models.CharField(
        max_length=500, verbose_name="Dataset description", blank=True, null=True
    )
    url = models.URLField(verbose_name="Dataset URL", blank=True, max_length=500)
    dimension = models.IntegerField(
        verbose_name="Dataset dimension (in MB)", blank=True, null=True
    )
    from_pdf = models.BooleanField(
        default=False, verbose_name="Dataset got from the PDF"
    )
    papers = models.ManyToManyField(
        Paper, related_name="datasets", verbose_name="Paper datasets"
    )
    last_update = models.DateTimeField(
        auto_now=True, verbose_name="Last update", blank=True, null=True
    )

    class Meta:
        verbose_name = "Dataset"
        verbose_name_plural = "Datasets"

    def __str__(self):
        return self.name


class PDFPaper(models.Model):

    paper = models.ForeignKey(
        Paper, on_delete=models.CASCADE, related_name="pdf_papers"
    )
    abstract = models.TextField(verbose_name="Abstract")
    supp_materials = models.TextField(
        verbose_name="Supplementary materials", blank=True, null=True
    )
    code_url = models.URLField(verbose_name="code URL", blank=True, max_length=500)
    text = models.TextField(verbose_name="Full paper text", blank=True, null=True)
    last_update = models.DateTimeField(
        auto_now=True, verbose_name="Last update", blank=True, null=True
    )

    class Meta:
        verbose_name = "PDF Paper"
        verbose_name_plural = "PDF Papers"

    def __str__(self):
        return self.title
