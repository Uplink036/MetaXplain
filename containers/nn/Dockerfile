FROM python
ENV DATABASE_URL=mongodb://localhost:27017/
WORKDIR /model
COPY . .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirments.txt
HEALTHCHECK CMD ./healthcheck.sh || bash -c 'kill -s 15 -1 && (sleep 10; kill -s 9 -1)'
CMD ["./run.sh"]