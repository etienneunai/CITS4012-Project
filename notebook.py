import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Readme
    *If there is something to be noted for the marker, please mention here.*

    *If you are planning to implement a program with Object Oriented Programming style, please put those the bottom of this ipynb file*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1.Dataset Processing
    (You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Model Implementation
    (You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3.Testing and Evaluation
    (You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
