#include <iostream>
#include <fstream>
#include <string>

using namespace std;

int main(){
  ifstream fin;
  fin.open("2(784dim).txt");
  if(!fin.is_open()){
    cout << "No input File read" << endl;
    return 0;
  }

  char c;
  string s;

  while(!fin.eof()){
    getline(fin, s);
    cout << s;
  }
  fin.close();
  return 0;
}
