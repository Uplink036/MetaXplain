FROM python
WORKDIR /metamorphical
COPY . .
RUN pip install -r requirments.txt
ENV DATABASE_URL=mongodb://localhost:27017/
CMD ["./run.sh"]